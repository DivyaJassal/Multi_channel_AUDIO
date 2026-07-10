import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

from config import load_config
from conv_tasnet import ConvTasNet
from utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="path to a mixed-speaker audio file")
    parser.add_argument("--out-dir", default="../outputs")
    parser.add_argument("--silence-threshold-db", type=float, default=-40.0,
                         help="output channels quieter than this (relative to the "
                              "loudest channel) are treated as unused padding and skipped")
    args = parser.parse_args()

    cfg = load_config()
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt.get("config", cfg)

    model = ConvTasNet(
        num_sources=model_cfg["max_sources"],
        enc_filters=model_cfg["enc_filters"],
        enc_kernel_size=model_cfg["enc_kernel_size"],
        bottleneck_channels=model_cfg["bottleneck_channels"],
        hidden_channels=model_cfg["hidden_channels"],
        kernel_size=model_cfg["tcn_kernel_size"],
        num_blocks=model_cfg["num_blocks"],
        num_repeats=model_cfg["num_repeats"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sample_rate = cfg["sample_rate"]
    audio, orig_sr = sf.read(args.input, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sample_rate)

    peak = np.max(np.abs(audio)) + 1e-8
    audio = audio / peak * 0.9

    mixture = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, T)

    with torch.no_grad():
        estimates = model(mixture)  # (1, max_sources, T)

    estimates = estimates.squeeze(0).cpu().numpy()

    # drop channels that are essentially silent — these correspond to the
    # zero-padded "no speaker here" slots the model learned during training
    channel_peaks = np.max(np.abs(estimates), axis=1)
    loudest = channel_peaks.max() + 1e-8
    channel_db = 20 * np.log10(channel_peaks / loudest + 1e-8)
    keep = channel_db > args.silence_threshold_db

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    n_written = 0
    for i, (src, keep_flag) in enumerate(zip(estimates, keep), start=1):
        if not keep_flag:
            continue
        n_written += 1
        out_path = out_dir / f"{stem}_spk{n_written}.wav"
        sf.write(out_path, src, sample_rate)
        print(f"wrote {out_path}")

    print(f"\nDetected ~{n_written} active speaker(s) out of "
          f"{model_cfg['max_sources']} model output channels.")


if __name__ == "__main__":
    main()