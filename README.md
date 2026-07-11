# Multi-Channel Speech Separation

Audio-only separation of overlapping, multi-speaker audio into individual per-speaker tracks — the "cocktail party problem" — built for mixtures of **2 to 4 concurrent speakers**.

Given a single audio file where multiple people are talking at once, this project outputs one clean audio track per speaker.

---

## Table of contents

- [Overview](#overview)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Usage](#usage)
- [Configuration reference](#configuration-reference)
- [Results](#results)
- [Performance notes](#performance-notes)
- [Possible extensions](#possible-extensions)

---

## Overview

This project trains a **Conv-TasNet** speech separation model using
**Permutation Invariant Training (PIT)** with an **SI-SDR** loss — the
standard, proven recipe for this task (the same family of techniques
behind WSJ0-mix / LibriMix baselines).

Key properties of this implementation:

- Handles a **variable number of real speakers per mixture** (2, 3, or 4),
  not just a single fixed count.
- Reports **separation quality broken down by speaker count**, so you can
  see exactly how quality degrades as more people talk at once.
- Runs on **CPU, NVIDIA CUDA, or Apple Silicon (MPS)** automatically.
- Caches decoded audio to disk so repeated training epochs don't re-decode
  FLAC files every time (a large real-world speedup — decoding, not model
  compute, is usually the actual bottleneck).

## How it works

**Model — Conv-TasNet** ([Luo & Mesgarani, 2019](https://arxiv.org/abs/1809.07454)):
instead of operating on a spectrogram, the model learns its own encoder
directly on the raw waveform. A stack of dilated 1D convolution blocks
(a Temporal Convolutional Network, or TCN) then predicts a mask per output
speaker channel; each mask is applied to the encoded mixture and decoded
back into a waveform.

**The variable-speaker-count problem:** a neural network needs a fixed
number of output channels, but real mixtures here have anywhere from 2 to
4 speakers. `dataset.py` handles this by always returning `max_sources`
(4) channels, zero-padding with silence whenever a mixture has fewer real
speakers.

**Why that needs a custom loss:** running standard PIT directly against
those zero-padded channels doesn't work — SI-SDR against an all-silent
target is mathematically undefined (zero target energy) and gives no
learning signal telling the model to output silence there. Instead,
`losses.pit_si_sdr_loss_variable`:
1. matches only the *real* speaker channels against the best-fitting
   subset of the model's output channels (Hungarian algorithm), and
2. separately pushes the leftover, unmatched output channels toward
   silence with a small energy penalty (`silence_weight` in
   `config.yaml`).

## Project structure

```
Multi_channel_AUDIO/
├── configs/
│   └── config.yaml            # all data / model / training hyperparameters
├── conversational_dataset_v2/ # dataset — not committed, see Dataset section
│   ├── train/sample_.../{metadata.json, mixture.flac, source_N.flac}
│   ├── val/...
│   └── test/...
├── src/
│   ├── config.py               # loads config.yaml
│   ├── dataset.py               # SpeechSeparationDataset (raw FLAC loading)
│   ├── dataloader.py            # create_dataloader() — non-cached
│   ├── cached_dataset.py        # caches decoded audio to disk after first read
│   ├── cached_dataloader.py     # create_dataloader() — cached (used by default)
│   ├── conv_tasnet.py           # the model
│   ├── losses.py                # SI-SDR + fixed-count and variable-count PIT
│   ├── utils.py                 # checkpointing, device selection, seeding
│   ├── trainer.py               # main training loop — RUN THIS TO TRAIN
│   ├── evaluate.py               # SI-SDRi report, broken down by speaker count
│   ├── separate.py               # run a trained model on a real audio file
│   └── test_dataset.py           # quick sanity check of the dataset loader
├── checkpoints/                # saved models (not committed)
├── reports/                    # evaluation CSV + chart output (not committed)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/DivyaJassal/Multi_channel_AUDIO.git
cd Multi_channel_AUDIO

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Dataset

This project uses a LibriMix-style dataset (derived from
[custom-sse-dataset](https://www.kaggle.com/datasets/paarthmanchanda/custom-sse-dataset)
on Kaggle), where every sample is a folder containing a mixture and its
isolated ground-truth speaker tracks:

```
conversational_dataset_v2/
├── train/
│   └── sample_0000001/
│       ├── metadata.json     # num_speakers, speaker_ids, overlap_ratio, ...
│       ├── mixture.flac
│       ├── source_1.flac
│       ├── source_2.flac
│       └── ... (up to source_4.flac)
├── val/
└── test/
```

Place this folder at the repo root, next to `src/` and `configs/`
(`configs/config.yaml`'s `data_root: "../conversational_dataset_v2"` assumes
scripts are run from inside `src/`). The dataset itself is **not**
committed to this repo — see `.gitignore`.

## Usage

Run all commands from inside `src/`.

**1. Sanity-check the dataset loader:**
```bash
cd src
python3 test_dataset.py
```
Expected output: sample counts per split and printed tensor shapes, no
errors.

**2. Train:**
```bash
python3 trainer.py
```
Watch the printed `val SI-SDR` each epoch. Checkpoints are written to
`../checkpoints/last.pt` (every epoch) and `../checkpoints/best.pt`
(whenever validation improves).

**3. Evaluate — generates a full report:**
```bash
python3 evaluate.py --checkpoint ../checkpoints/best.pt --split test
```
Prints, and saves to `../reports/`:
- `results_test.csv` — SI-SDR / SI-SDRi per speaker count
- `results_test.png` — bar chart of separation quality vs. speaker count

**4. Separate a real audio file:**
```bash
python3 separate.py --checkpoint ../checkpoints/best.pt --input /path/to/recording.wav
```
Writes one `.wav` per detected active speaker to `../outputs/`.

## Configuration reference

All hyperparameters live in `configs/config.yaml`:

| Key | Meaning |
|---|---|
| `sample_rate` | audio sample rate in Hz (16000 for this dataset) |
| `segment_seconds` | random crop length used during training |
| `max_sources` | fixed number of model output channels (real speaker counts ≤ this) |
| `data_root` | path to the dataset folder |
| `train_batch_size` / `val_batch_size` | batch sizes |
| `num_workers` | parallel data-loading processes |
| `enc_filters`, `enc_kernel_size`, `bottleneck_channels`, `hidden_channels`, `tcn_kernel_size`, `num_blocks`, `num_repeats` | Conv-TasNet architecture size |
| `epochs`, `lr`, `grad_clip_norm` | training hyperparameters |
| `silence_weight` | weight of the silence penalty for unmatched/padded output channels |
| `checkpoint_dir` | where model checkpoints are saved |

## Results

*Fill in after your training run — `evaluate.py` generates these numbers
and the chart automatically.*

| Speakers | Input SI-SDR (dB) | Output SI-SDR (dB) | SI-SDRi (dB) |
|---|---|---|---|
| 2 | | | |
| 3 | | | |
| 4 | | | |

![Separation quality vs speaker count](reports/results_test.png)

## Performance notes

- **Device:** automatically uses CUDA, then Apple Silicon (MPS), then
  falls back to CPU.
- **Audio caching:** the first training epoch decodes FLAC files as usual
  and writes a `cache.pt` next to each sample; every epoch after that
  loads the cached tensor directly, skipping FLAC decoding entirely. This
  was, in practice, a far bigger speedup than any model-size or batch-size
  change, since audio decoding — not GPU compute — was the actual
  bottleneck.
- If disk space is a concern, `cache.pt` files can be safely deleted at
  any time; they'll simply be regenerated on the next read.

## Possible extensions

- Train one model per fixed speaker count instead of a single padded
  model, for a cleaner per-count comparison.
- Swap Conv-TasNet for a stronger separator (SepFormer, DPRNN) for a
  quality bump at the cost of more compute.
- Recursive/iterative separation to handle an unbounded, unknown number
  of speakers rather than a fixed maximum.