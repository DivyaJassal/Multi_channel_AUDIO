from pathlib import Path
import csv

import torch
from tqdm import tqdm

from config import load_config
from dataloader import create_dataloader
from conv_tasnet import ConvTasNet
from losses import pit_si_sdr_loss_variable
from utils import (
    set_seed,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    get_device,
    AverageMeter,
)


def run_epoch(model, loader, optimizer, device, cfg, train=True):
    """
    Runs one training or validation epoch.
    """

    if train:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter()
    sisdr_meter = AverageMeter()

    with torch.set_grad_enabled(train):

        for batch in tqdm(
            loader,
            desc="train" if train else "val",
            leave=False,
        ):

            mixture = batch["mixture"].to(device)
            sources = batch["sources"].to(device)
            num_sources = batch["num_sources"].to(device)

            estimates = model(mixture)

            loss, best_sisdr = pit_si_sdr_loss_variable(
                estimates,
                sources,
                num_sources,
                silence_weight=cfg["silence_weight"],
            )

            if train:

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg["grad_clip_norm"],
                )

                optimizer.step()

            batch_size = mixture.size(0)

            loss_meter.update(loss.item(), batch_size)

            sisdr_meter.update(
                best_sisdr.mean().item(),
                batch_size,
            )

    return loss_meter.avg, sisdr_meter.avg


def main():

    cfg = load_config()

    set_seed(cfg["seed"])

    device = get_device()

    print("=" * 70)
    print(f"Using device : {device}")
    print("=" * 70)

    train_loader = create_dataloader(
        root_dir=cfg["data_root"],
        split="train",
        sample_rate=cfg["sample_rate"],
        segment_seconds=cfg["segment_seconds"],
        max_sources=cfg["max_sources"],
        batch_size=cfg["train_batch_size"],
        shuffle=cfg["shuffle"],
        num_workers=cfg["num_workers"],
    )

    val_loader = create_dataloader(
        root_dir=cfg["data_root"],
        split="val",
        sample_rate=cfg["sample_rate"],
        segment_seconds=cfg["segment_seconds"],
        max_sources=cfg["max_sources"],
        batch_size=cfg["val_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    model = ConvTasNet(
        num_sources=cfg["max_sources"],
        enc_filters=cfg["enc_filters"],
        enc_kernel_size=cfg["enc_kernel_size"],
        bottleneck_channels=cfg["bottleneck_channels"],
        hidden_channels=cfg["hidden_channels"],
        kernel_size=cfg["tcn_kernel_size"],
        num_blocks=cfg["num_blocks"],
        num_repeats=cfg["num_repeats"],
    ).to(device)

    print(
        f"Model Parameters : {count_parameters(model):,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    ckpt_dir = Path(cfg["checkpoint_dir"])

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_ckpt = ckpt_dir / "last.pt"

    start_epoch = 1

    best_val_sisdr = -1e9

    if last_ckpt.exists():

        print("\nFound existing checkpoint.")

        ckpt = load_checkpoint(
            last_ckpt,
            model,
            optimizer,
            device=device,
        )

        start_epoch = ckpt["epoch"] + 1

        best_val_sisdr = ckpt["best_val_sisdr"]

        print(
            f"Resuming from Epoch {start_epoch}"
        )

        print(
            f"Best Validation SI-SDR : "
            f"{best_val_sisdr:.2f} dB"
        )

    else:

        print("\nNo checkpoint found.")

        print("Starting fresh training.")

    runs_dir = Path("../runs")

    runs_dir.mkdir(
        exist_ok=True,
    )

    log_file = runs_dir / "train_log.csv"

    if not log_file.exists():

        with open(log_file, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_sisdr",
                    "val_sisdr",
                    "learning_rate",
                ]
            )

    patience = 8

    patience_counter = 0
    model = ConvTasNet(
        num_sources=cfg["max_sources"],
        enc_filters=cfg["enc_filters"],
        enc_kernel_size=cfg["enc_kernel_size"],
        bottleneck_channels=cfg["bottleneck_channels"],
        hidden_channels=cfg["hidden_channels"],
        kernel_size=cfg["tcn_kernel_size"],
        num_blocks=cfg["num_blocks"],
        num_repeats=cfg["num_repeats"],
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # Resume training automatically if last.pt exists
    # ----------------------------------------------------
    start_epoch = 1
    best_val_sisdr = -1e9

    last_ckpt = ckpt_dir / "last.pt"

    if last_ckpt.exists():

        print(f"Resuming from {last_ckpt}")

        checkpoint = load_checkpoint(
            last_ckpt,
            model,
            optimizer,
            device
        )

        start_epoch = checkpoint["epoch"] + 1
        best_val_sisdr = checkpoint["best_val_sisdr"]

        print(f"Continuing from epoch {start_epoch}")

    else:

        print("Starting fresh training")

    # ----------------------------------------------------
    # Training Loop
    # ----------------------------------------------------
    for epoch in range(start_epoch, cfg["epochs"] + 1):

        train_loss, train_sisdr = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg,
            train=True
        )

        val_loss, val_sisdr = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            cfg,
            train=False
        )

        scheduler.step(val_sisdr)

        print(
            f"Epoch {epoch:03d} | "
            f"Train SI-SDR: {train_sisdr:.2f} dB | "
            f"Val SI-SDR: {val_sisdr:.2f} dB | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_val_sisdr,
            extra={"config": cfg}
        )

        if val_sisdr > best_val_sisdr:

            best_val_sisdr = val_sisdr

            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_val_sisdr,
                extra={"config": cfg}
            )

            print(
                f"New Best Model Saved "
                f"(Val SI-SDR = {val_sisdr:.2f} dB)"
            )
if __name__ == "__main__":
    main()