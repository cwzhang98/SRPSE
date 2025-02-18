import math
import os
import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II
from fairseq.data.data_utils import lengths_to_padding_mask, compute_mask_indices
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model
)
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer
)
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.models.wav2vec import Wav2Vec2Model


@dataclass
class XSTNetConfig(FairseqDataclass):
    w2v_model_path: str = field(
        default='', metadata={"help": "path to wav2vec 2.0 model"}
    )
    freeze_w2v: bool = field(default=True, metadata={"help": "freeze acoustic encoder"})
    # transformer
    activation_fn: str = field(
        default='gelu',
        metadata={"help": "activation function to use"}
    )
    max_source_positions: int = II("task.max_source_positions")
    # dropout
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability"}
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability after activation in FFN"}
    )
    # transformer encoder
    encoder_layers: int = field(
        default=6,
        metadata={"help": "num of encoder layers"}
    )
    encoder_embed_dim: int = field(
        default=512,
        metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=2048,
        metadata={"help": "encoder embedding dimension"}
    )
    encoder_attention_heads: int = field(
        default=8,
        metadata={"help": "num encoder attention heads"}
    )
    encoder_normalize_before: bool = field(
        default=True,
        metadata={"help": "apply layernorm before each encoder block (pre-norm)"}
    )
    no_positional_embeddings: bool = field(
        default=False,
        metadata={"help": "disable positional embedding"}
    )
    encoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "learnable positional embedding"}
    )
    textual_encoder_hidden_state: Optional[str] = field(
        default="",
        metadata={"help": "comma separated index, specify which layer's hidden state returned in EncoderOut"}
    )
    # transformer decoder
    decoder_layers: int = field(
        default=6,
        metadata={"help": "num of decoder layers"}
    )
    decoder_embed_dim: int = field(
        default=512,
        metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048,
        metadata={"help": "decoder embedding dimension"}
    )
    decoder_attention_heads: int = field(
        default=8,
        metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=True,
        metadata={"help": "apply layernorm before each decoder block (pre-norm)"}
    )
    share_decoder_input_output_embed: bool = field(
        default=True,
        metadata={"help": "share decoder input and output embeddings"}
    )
    layernorm_embedding: bool = field(
        default=False,
        metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False,
        metadata={"help": "if True, dont scale embeddings"}
    )


@register_model("xstnet", dataclass=XSTNetConfig)
class XSTNet(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.is_audio_input = True
        self.num_updates = 0

    @classmethod
    def build_model(cls, cfg: XSTNetConfig, task):
        encoder_embed_tokens = cls.build_embedding(task.target_dictionary, cfg.encoder_embed_dim)
        encoder = cls.build_encoder(cfg, task.target_dictionary, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, task.target_dictionary, encoder_embed_tokens)
        return cls(encoder, decoder)

    @classmethod
    def build_embedding(cls, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        return emb

    @classmethod
    def build_encoder(cls, cfg, dict, embed_tokens):
        encoder = XSTNetEncoder(cfg, dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoder(cfg, tgt_dict, embed_tokens)

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            is_audio_input=True,
            **kwargs
    ):
        encoder_out = self.encoder(src_tokens, src_lengths, is_audio_input)
        decoder_out = self.decoder(prev_output_tokens=prev_output_tokens,
                                   encoder_out=encoder_out)
        if self.training:
            return decoder_out, encoder_out
        return decoder_out

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class XSTNetEncoder(FairseqEncoder):
    def __init__(self, cfg: XSTNetConfig, dictionary, src_embedding):
        super().__init__(dictionary)
        self.cfg = cfg
        self.freeze_w2v = cfg.freeze_w2v
        self.embed_tokens = src_embedding
        # module_name：assign name for call FairseqDropout.make_generation_fast_ func to keep dropout during inference
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.num_updates = 0
        self.padding_idx = dictionary.pad()
        self.shared_encoder_embed_dim = src_embedding.embedding_dim
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(self.shared_encoder_embed_dim)
        self.w2v_model, self.subsample_audio, self.ctc_projection = None, None, None

        self.build_acoustic_encoder(cfg)
        self.build_shared_encoder(cfg)

    def build_acoustic_encoder(self, cfg: XSTNetConfig):
        assert cfg.w2v_model_path is not None and os.path.isfile(cfg.w2v_model_path)
        # load checkpoints
        ckpt = torch.load(cfg.w2v_model_path)
        w2v_args = convert_namespace_to_omegaconf(ckpt["args"])
        self.w2v_model = Wav2Vec2Model.build_model(w2v_args.model, task=None)
        self.subsample_audio = Conv1dSubsampler(
            ckpt["args"].encoder_embed_dim,  # 768
            1024,
            self.shared_encoder_embed_dim,
            [5, 5],
        )
        self.ctc_projection = nn.Linear(
            768,
            self.embed_tokens.weight.shape[0],
        )
        self.w2v_model.load_state_dict(ckpt["model"], strict=True)

    def build_shared_encoder(self, cfg: XSTNetConfig):
        self.positional_embed = (
            PositionalEmbedding(
                cfg.max_source_positions,
                self.shared_encoder_embed_dim,
                self.padding_idx,
                cfg.encoder_learned_pos
            )
            if not cfg.no_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.embedding_layernorm = LayerNorm(self.shared_encoder_embed_dim)
        else:
            self.embedding_layernorm = None

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(cfg.encoder_layers)]
        )
        if cfg.encoder_normalize_before:
            self.layer_norm = LayerNorm(self.shared_encoder_embed_dim)
        else:
            self.layer_norm = None

    def encode_audio(self, src_tokens, src_lengths):
        padding_mask = lengths_to_padding_mask(src_lengths)
        if self.cfg.freeze_w2v:
            with torch.no_grad():
                w2v_feature, padding_mask = self.w2v_model.extract_features(src_tokens, padding_mask)
        else:
            w2v_feature, padding_mask = self.w2v_model.extract_features(src_tokens, padding_mask)
        if padding_mask is None:
            padding_mask = torch.zeros(
                (w2v_feature.shape[0], w2v_feature.shape[1]), device=w2v_feature.device)
        ctc_logits, ctc_padding_mask = self.ctc_projection(self.dropout_module(w2v_feature)).float(), padding_mask
        feature_lengths = (1 - padding_mask.int()).sum(dim=1)
        # cnn subsampling
        w2v_feature, feature_lengths = self.subsample_audio(w2v_feature, feature_lengths)  # T x B x C

        w2v_feature *= self.embed_scale
        encoder_padding_mask = lengths_to_padding_mask(feature_lengths)
        if self.positional_embed is not None:
            positions = self.positional_embed(encoder_padding_mask).transpose(0, 1)  # T x B x C
            w2v_feature += positions

        w2v_feature = self.dropout_module(w2v_feature)
        return w2v_feature, encoder_padding_mask, ctc_logits, ctc_padding_mask

    def encode_text(self, src_tokens):
        embedding = self.embed_tokens(src_tokens)
        x = self.embed_scale * embedding
        if self.positional_embed is not None:
            x += self.positional_embed(src_tokens)
        if self.embedding_layernorm is not None:
            x = self.embedding_layernorm(x)
        x = self.dropout_module(x)
        x = x.transpose(0, 1)  # B x T x C -> T x B x C
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        return x, encoder_padding_mask

    def forward(
            self,
            src_tokens,
            src_lengths,
            is_audio_input=True,
            **kwargs
    ):
        """
            src_tokens: (B, T)
            src_lengths:(B)
        """
        if is_audio_input:  # forward audio
            x, encoder_padding_mask, ctc_logits, ctc_padding_mask = self.encode_audio(src_tokens, src_lengths)
            speech_repr = x.clone()
        else:  # forward text
            x, encoder_padding_mask = self.encode_text(src_tokens)
            text_repr = x.clone()
        encoder_states = []
        for index, layer in enumerate(self.transformer_layers):
            x = layer(x, encoder_padding_mask)
            encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "ctc_logits": ctc_logits if is_audio_input else None,
            "ctc_padding_mask": ctc_padding_mask if is_audio_input else None,
            "encoder_embedding": None,  # T x B x C
            "encoder_states": encoder_states,
            "src_tokens": None,
            "src_lengths": None,
            "speech_repr": speech_repr if is_audio_input else None,
            "text_repr": text_repr if not is_audio_input else None,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )
        return {
            "encoder_out": new_encoder_out,
            "encoder_padding_mask": new_encoder_padding_mask,
            "encoder_embedding": encoder_out["encoder_embedding"],
            "encoder_states": None,
            "src_tokens": None,
            "src_lengths": None
        }
