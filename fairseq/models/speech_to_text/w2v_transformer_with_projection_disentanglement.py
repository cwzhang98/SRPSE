import copy
import os
import torch
from torch import nn, Tensor
from dataclasses import dataclass, field
from typing import Optional, Any
from torch.autograd import Function
import numpy as np

from fairseq.data.data_utils import lengths_to_padding_mask, compute_mask_indices
from fairseq.models import register_model

from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer
)
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler
from fairseq.models.wav2vec import Wav2Vec2Model
from fairseq.models.speech_to_text.w2v_transformer_with_feat_fusion import (
    W2vTransformerWithFeatFusionConfig,
    W2vTransformerWithFeatFusionModel,
    W2vTransformerWithFeatFusionEncoder
)


@dataclass
class W2vTransformerWithProjectionDisentanglementConfig(W2vTransformerWithFeatFusionConfig):
    non_content_encoder_layers: Optional[int] = field(default=3)
    mt_ckpt_path: Optional[str] = field(default=None)


@register_model("w2v_transformer_with_projection_disentanglement",
                dataclass=W2vTransformerWithProjectionDisentanglementConfig)
class W2vTransformerWithProjectionDisentanglementModel(W2vTransformerWithFeatFusionModel):
    @classmethod
    def build_model(cls, cfg: W2vTransformerWithProjectionDisentanglementConfig, task):
        encoder_embed_tokens = cls.build_embedding(task.target_dictionary, cfg.encoder_embed_dim)
        if cfg.mt_ckpt_path is not None:
            # load embedding weights
            mt_ckpt = torch.load(cfg.mt_ckpt_path)
            encoder_embed_tokens.weight.data = mt_ckpt["model"]["encoder.embed_tokens.weight"]
        encoder = cls.build_encoder(cfg, task.target_dictionary, encoder_embed_tokens, len(task.speaker_to_id))
        decoder = cls.build_decoder(cfg, task.target_dictionary, encoder_embed_tokens)
        if cfg.mt_ckpt_path is not None:
            for n, p in encoder.content_encoders.named_parameters():
                p.data = mt_ckpt["model"][f"encoder.layers.{n}"]
            for n, p in encoder.transformer_layers.named_parameters():
                l, k = n.split(".", maxsplit=1)
                n = str(int(l) + 1) + "." + k
                p.data = mt_ckpt["model"][f"encoder.layers.{n}"]
            for n, p in decoder.named_parameters():
                if n != "embed_tokens.weight":
                    p.data = mt_ckpt["model"][f"decoder.{n}"]

        return cls(encoder, decoder)

    @classmethod
    def build_encoder(cls, cfg, dict, embed_tokens, speaker_to_id):
        encoder = W2vTransformerWithProjectionDisentanglementEncoder(cfg, dict, embed_tokens, speaker_to_id)
        return encoder

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


class W2vTransformerWithProjectionDisentanglementEncoder(W2vTransformerWithFeatFusionEncoder):
    def __init__(
            self,
            cfg: W2vTransformerWithProjectionDisentanglementConfig,
            dictionary,
            src_embedding,
            num_speakers
    ):
        super().__init__(cfg, dictionary, src_embedding)
        self.cfg = cfg
        self.speaker_classifier = SpeakerClassifier(cfg.encoder_embed_dim, num_speakers)
        self.noise_classifier = NoiseClassifier(cfg.encoder_embed_dim)
        self.num_updates = 0

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def build_acoustic_encoder(self, cfg: W2vTransformerWithProjectionDisentanglementConfig):
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

    def build_shared_encoder(self, cfg: W2vTransformerWithProjectionDisentanglementConfig):
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
            [TransformerEncoderLayer(cfg) for _ in range(cfg.encoder_layers - cfg.non_content_encoder_layers)]
        )
        self.content_encoders = nn.ModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(cfg.non_content_encoder_layers)]
        )
        self.non_content_encoders = nn.ModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(cfg.non_content_encoder_layers)]
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
        else:  # forward text
            x, encoder_padding_mask = self.encode_text(src_tokens)

        if is_audio_input:
            non_content_feats = x.clone()
            for layer in self.non_content_encoders:
                non_content_feats = layer(non_content_feats, encoder_padding_mask)
            for layer in self.content_encoders:
                x = layer(x, encoder_padding_mask)
            ci_feats = x.clone()
            speaker_logits = self.speaker_classifier(non_content_feats.clone(),
                                                     encoder_padding_mask) if self.training else None
            noise_logits = self.noise_classifier(non_content_feats.clone(),
                                                 encoder_padding_mask) if self.training else None
            proj = self.projection(x, non_content_feats)
            x = x - proj
            purified_feats = x.clone()  # clone for consistency loss computing

            for layer in self.transformer_layers:
                x = layer(x, encoder_padding_mask)
        else:
            for layer in self.content_encoders:
                x = layer(x, encoder_padding_mask)
            text_repr = x.clone()
            for layer in self.transformer_layers:
                x = layer(x, encoder_padding_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "speaker_logits": speaker_logits if is_audio_input else None,
            "ctc_logits": ctc_logits if is_audio_input else None,
            "ctc_padding_mask": ctc_padding_mask if is_audio_input else None,
            "noise_logits": noise_logits if is_audio_input else None,
            "purified_feats": purified_feats if is_audio_input else None,
            "non_content_feats": non_content_feats if is_audio_input else None,
            "ci_feats": ci_feats if is_audio_input else None,
            "text_repr": text_repr if not is_audio_input else None,
            "encoder_embedding": None,  # T x B x C
            "encoder_states": None,
            "src_tokens": None,
            "src_lengths": None,
            "shared_encoder_states": None,
        }

    def projection(self, x, y):
        """
            shape: T x B x C
            project x to y: [(x dot y) / norm(y)] * [y / norm(y)]
        """
        y = y.float()
        unit_vector = nn.functional.normalize(y, p=2, dim=-1)
        projection = (x * unit_vector).sum(dim=-1).unsqueeze(-1) * unit_vector
        return projection


class SpeakerClassifier(nn.Module):
    def __init__(self, model_dim, num_speakers):
        super().__init__()
        self.num_speakers = num_speakers
        self.projections = nn.Sequential(
            nn.Linear(model_dim, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, num_speakers)
        )

    def forward(self, x: Tensor, padding_mask):
        x = x.transpose(0, 1).masked_fill_(padding_mask.unsqueeze(-1), 0.0)  # T x B x C -> B x T x C
        x = x.mean(dim=1)  # B x C
        x = self.projections(x)
        return x


class NoiseClassifier(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.num_class = 5
        self.projections = nn.Sequential(
            nn.Linear(model_dim, 1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1024, self.num_class)
        )

    def forward(self, x: Tensor, padding_mask):  # T x B x C
        x = x.transpose(0, 1).masked_fill_(padding_mask.unsqueeze(-1), 0.0)
        x = x.mean(dim=1)  # B x C
        x = self.projections(x)
        return x


class GradientReverse(Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        return grad_output.neg() * ctx.coeff, None


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha, low, high, max_updates):
        super().__init__()
        self.alpha = alpha
        self.low = low
        self.high = high
        self.max_updates = max_updates

    def forward(self, x, num_updates):
        coeff = np.float64(
            2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * num_updates / self.max_updates))
            - (self.high - self.low) + self.low
        )
        return GradientReverse.apply(x, coeff)
