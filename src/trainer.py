from pathlib import Path

import torch
from tqdm import tqdm

from config import load_config
from dataloader import create_dataloader
from conv_tasnet import ConvTasNet
from losses import pit_si_sdr_loss_variable, si_sdr
from utils import set_seed, count_parameters, save_checkpoint, get_device, AverageMeter


def run_epoch(model, loader, optimizer, device, cfg, train=True):
    model.train(mode=train)
    loss_meter = AverageMeter()
    sisdr_meter = AverageMeter()

    torch.set_grad_enabled(train)
    for batch in tqdm(loader, desc="train" if train else "val", leave=False):
        mixture = batch["mixture"].to(device)          # (B, 1, T)
        sources = batch["sources"].to(device)           # (B, max_sources, T)
        num_sources = batch["num_sources"].to(device)    # (B,) real speaker count

        estimates = model(mixture)                      # (B, max_sources, T)

        loss, best_sisdr = pit_si_sdr_loss_variable(
            estimates, sources, num_sources,
            silence_weight=cfg["silence_weight"],
        )

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            optimizer.step()

        batch_size = mixture.shape[0]
        loss_meter.update(loss.item(), batch_size)
        sisdr_meter.update(best_sisdr.mean().item(), batch_size)

    return loss_meter.avg, sisdr_meter.avg


def main():
    cfg = load_config()
    set_seed(cfg.get("seed", 0))
    device = get_device()
    print(f"Using device: {device}")

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
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_sisdr = -1e9
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_sisdr = run_epoch(model, train_loader, optimizer, device, cfg, train=True)
        val_loss, val_sisdr = run_epoch(model, val_loader, optimizer, device, cfg, train=False)
        scheduler.step(val_sisdr)

        print(f"epoch {epoch:3d} | train SI-SDR {train_sisdr:6.2f} dB "
              f"| val SI-SDR {val_sisdr:6.2f} dB "
              f"| lr {optimizer.param_groups[0]['lr']:.2e}")

        save_checkpoint(
            ckpt_dir / "last.pt", model, optimizer, epoch, best_val_sisdr,
            extra={"config": cfg},
        )

        if val_sisdr > best_val_sisdr:
            best_val_sisdr = val_sisdr
            save_checkpoint(
                ckpt_dir / "best.pt", model, optimizer, epoch, best_val_sisdr,
                extra={"config": cfg},
            )
            print(f"  -> new best model saved (val SI-SDR {val_sisdr:.2f} dB)")


if __name__ == "__main__":
    main()