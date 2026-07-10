import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from config import load_config
from dataloader import create_dataloader
from conv_tasnet import ConvTasNet
from losses import pit_si_sdr_loss_variable, si_sdr
from utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--report-dir", default="../reports",
                         help="where to save results.csv and results.png")
    args = parser.parse_args()

    cfg = load_config()
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt.get("config", cfg)  # prefer the config saved with the checkpoint

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

    loader = create_dataloader(
        root_dir=cfg["data_root"],
        split=args.split,
        sample_rate=cfg["sample_rate"],
        segment_seconds=cfg["segment_seconds"],
        max_sources=cfg["max_sources"],
        batch_size=1,  # per-example, so per-sample num_sources is unambiguous
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    # bucket results by the REAL number of speakers in each mixture, so you
    # get the quality-vs-concurrent-speaker-count breakdown for evaluation
    per_count_out = defaultdict(list)
    per_count_in = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluating [{args.split}]"):
            mixture = batch["mixture"].to(device)
            sources = batch["sources"].to(device)
            num_sources = batch["num_sources"].to(device)
            c = int(num_sources[0].item())

            estimates = model(mixture)
            _, best_sisdr = pit_si_sdr_loss_variable(estimates, sources, num_sources)

            mixture_rep = mixture.squeeze(1).unsqueeze(1).expand(-1, c, -1)
            input_sisdr = si_sdr(mixture_rep, sources[:, :c]).mean().item()

            per_count_out[c].append(best_sisdr.item())
            per_count_in[c].append(input_sisdr)

    print(f"\n=== Results on '{args.split}' split, broken down by speaker count ===")
    rows = []
    for c in sorted(per_count_out.keys()):
        out_vals = per_count_out[c]
        in_vals = per_count_in[c]
        avg_out = sum(out_vals) / len(out_vals)
        avg_in = sum(in_vals) / len(in_vals)
        sisdri = avg_out - avg_in
        print(f"{c} speakers | n={len(out_vals):4d} | "
              f"input SI-SDR={avg_in:6.2f} dB | output SI-SDR={avg_out:6.2f} dB | "
              f"SI-SDRi={sisdri:6.2f} dB")
        rows.append({
            "num_speakers": c, "n_examples": len(out_vals),
            "input_sisdr_db": round(avg_in, 3), "output_sisdr_db": round(avg_out, 3),
            "sisdri_db": round(sisdri, 3),
        })

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / f"results_{args.split}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["num_speakers", "n_examples",
                                                "input_sisdr_db", "output_sisdr_db", "sisdri_db"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = [r["num_speakers"] for r in rows]
        sisdri = [r["sisdri_db"] for r in rows]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar([str(c) for c in counts], sisdri, color="#4C72B0")
        ax.set_xlabel("Number of concurrent speakers")
        ax.set_ylabel("SI-SDR improvement (dB)")
        ax.set_title(f"Separation quality vs. speaker count ({args.split} split)")
        ax.bar_label(bars, fmt="%.1f")
        fig.tight_layout()

        png_path = report_dir / f"results_{args.split}.png"
        fig.savefig(png_path, dpi=150)
        print(f"Saved chart: {png_path}")
    except ImportError:
        print("matplotlib not installed — skipping chart (CSV was still saved). "
              "Run: pip install matplotlib")


if __name__ == "__main__":
    main()