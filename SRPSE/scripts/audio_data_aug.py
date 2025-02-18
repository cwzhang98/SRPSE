import torch
import torchaudio
import torchaudio.transforms as T
from torch import nn
from fairseq.data.audio.audio_utils import parse_path
import csv
from tqdm import tqdm
import argparse
import pandas as pd


class AudioPerturbationPipeline(nn.Module):
    PITCH_SHIFT = [-1, 0, 1]
    BACKGROUND_NOISE = [5, 10, 20, 50, 100]
    TIME_STRETCH = [0.8, 0.9, 1.1, 1.2]

    def __init__(self, pitch_idx, noise_idx, stretch_idx, sample_rate=16000):
        super().__init__()
        self.pitch_idx = pitch_idx
        self.noise_idx = noise_idx
        self.stretch_idx = stretch_idx
        self.stretch_rate = self.TIME_STRETCH[self.stretch_idx]
        self.shift = T.PitchShift(sample_rate, self.PITCH_SHIFT[self.pitch_idx]).cuda(0)
        self.snr = self.BACKGROUND_NOISE[self.noise_idx]
        self.stretch = T.TimeStretch(fixed_rate=self.stretch_rate, n_freq=257)
        self.spectrogram = T.Spectrogram(n_fft=512, power=None)
        self.invert = T.InverseSpectrogram(n_fft=512)

    def forward(self, waveform):
        #  改变音高
        if self.PITCH_SHIFT[self.pitch_idx] != 0:
            self.shift = self.shift.cuda()
            waveform = self.shift(waveform)
        # 添加高斯噪声
        if self.snr != 100:
            noise = torch.randn(waveform.size()).cuda()
            scale = torch.sqrt(torch.mean(waveform ** 2)) / (torch.sqrt(torch.mean(noise ** 2)) * 10 ** (self.snr / 10)).cuda()
            waveform = waveform + scale * noise
        # 频谱增强
        self.spectrogram = self.spectrogram.cuda()
        self.stretch = self.stretch.cuda()
        self.invert = self.invert.cuda()
        spec = self.spectrogram(waveform)
        spec = self.stretch(spec, self.stretch_rate)
        waveform = self.invert(spec)
        return waveform


def process(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # 读tsv
    with open(args.tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t", quotechar=None,
                                doublequote=False, lineterminator="\n",
                                quoting=csv.QUOTE_NONE)
        samples = [dict(e) for e in reader]
    for i, sample in tqdm(enumerate(samples)):
        # load audio
        _path, slice_ptr = parse_path(samples[i]["audio"])
        waveform, sample_rate = torchaudio.load(_path, slice_ptr[0], slice_ptr[1])
        # augment audio
        pitch = torch.randint(low=0, high=3, size=(1,))
        noise = torch.randint(low=0, high=5, size=(1,))
        stretch = torch.randint(low=0, high=4, size=(1,))
        pitch, noise, stretch = pitch.item(), noise.item(), stretch.item()
        perturb = AudioPerturbationPipeline(pitch, noise, stretch)
        waveform = waveform.cuda()
        waveform_aug = perturb(waveform)
        # save audio
        torchaudio.save(args.aug_dir + f"/aug_{i}.wav", waveform_aug.cpu(), sample_rate)
        # write dict
        samples[i]["pitch"], samples[i]["noise"] = pitch, noise
        samples[i]["audio_aug"] = args.aug_dir + f"/aug_{i}.wav"

    MANIFEST_COLUMNS = ["id", "audio", "n_frames", "speaker", "tgt_text", "src_text", "src_lang", "tgt_lang",
                        "audio_aug", "pitch", "noise"]
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    for sample in samples:
        for col in MANIFEST_COLUMNS:
            manifest[col].append(sample.get(col, ""))
    df = pd.DataFrame.from_dict(manifest)
    df.to_csv(
        args.aug_tsv_path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv-path", required=True, type=str)
    parser.add_argument("--aug-tsv-path", required=True, type=str)
    parser.add_argument("--aug-dir", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()

