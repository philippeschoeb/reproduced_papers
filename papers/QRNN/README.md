# QRNN — Quantum Recurrent Neural Networks (Neural Networks, 2023)

This project bootstraps a reproduction of the paper on quantum recurrent neural networks (QRNN, Neural Networks, 2023). It currently ships a classical RNN baseline over weather time-series data and a scaffold for extending the work toward the quantum architecture.

## Reference and Attribution

- Paper: Quantum Recurrent Neural Networks for sequential learning (Neural Networks, 2023)
- Authors: Yanan Li, Zhimin Wang, Rongbing Han, Shangshang Shi, Jiaxin Li, Ruimin Shang, Haiyong Zheng, Guoqiang Zhong, Yongjian Gu
- DOI/ArXiv: https://doi.org/10.1016/j.neunet.2023.07.003 (publisher page: https://www.sciencedirect.com/science/article/abs/pii/S089360802300360X)
- Original repository (if any): not referenced here
- License and attribution notes: cite the published Neural Networks article when using results derived from this code.

## Overview

The reproduction is staged:

- **Stage 1 (implemented here):** classical RNN baseline for sequence forecasting on a meteorological dataset.
- **Stage 2:** swap in the QRNN architecture described in the paper and compare against the baseline metrics.

Defaults target the Kaggle Szeged weather dataset (`budincsevity/szeged-weather`) with a preprocessing step that aggregates daily statistics.

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Use the repository-level runner (`implementation.py` at the repo root) with `--paper QRNN`.

```bash
# From the repo root
python implementation.py --paper QRNN --help

# From inside this folder
python ../../implementation.py --help
```

Common options (see `cli.json` for the full schema alongside the global flags injected by the shared runner):

- `--config PATH` Load an additional JSON config (merged over defaults).
- `--epochs INT` Override `training.epochs`.
- `--batch-size INT` Override `dataset.batch_size`.
- `--sequence-length INT` History length used for forecasting.
- `--hidden-dim INT` RNN hidden dimension.
- Standard global flags: `--seed`, `--dtype`, `--device`, `--log-level`, `--outdir`.

Example runs:

```bash
# Quick smoke test (no download; uses a synthetic generator from papers/shared/time_series)
python implementation.py --paper QRNN --config configs/damped_shm_default_rnn.json --epochs 1

# Same run from inside QRNN/
python ../../implementation.py --config configs/damped_shm_default_rnn.json --epochs 1

# Train on the Szeged Kaggle dataset (downloads on first run if needed)
python implementation.py --paper QRNN --outdir runs/qrnn_baseline
```

### Dataset setup & preprocessing

- Datasets are stored under the shared data root: by default `data/QRNN/` at the repo root. You can override this location with `DATA_DIR=/abs/path` or `python implementation.py --data-root /abs/path ...`.
- The default configuration uses `dataset.path="szeged-weather.csv"` and `dataset.preprocess="szeged_weather"`. If the raw file is missing, it will be downloaded from `budincsevity/szeged-weather`, then preprocessed into `szeged-weather.preprocess.csv` the first time. Subsequent runs reuse the preprocessed file.
- The Szeged-specific preprocessor consumes `Formatted Date`, `Temperature (C)`, `Humidity`, `Wind Speed (km/h)`, and `Pressure (millibars)` and outputs a daily CSV with columns: `date`, `min_temperature`, `max_temperature`, `avg_humidity`, `avg_wind_speed`, `avg_pressure`.
- To run without downloading any CSV, use `configs/damped_shm_default_rnn.json`, which generates a synthetic time-series via the shared time-series generators (`papers/shared/time_series/`).
- Additional dataset-specific preprocessors can be registered in `lib/preprocess.py`; set `dataset.preprocess` to the corresponding key to enable them. If a `<file>.preprocess.csv` already exists, it is reused without rebuilding.
- Use `dataset.max_rows` to cap the number of rows ingested from the CSV (applied before splitting). Normalization stats are computed on the resulting training split to avoid leakage.

### Outputs

Each run writes to `<outdir>/run_YYYYMMDD-HHMMSS/` and includes:

- `config_snapshot.json` — resolved configuration used for the run
- `metrics.json` — train/validation loss history
- `predictions.csv` — reference vs model predictions for train/val/test splits
- `metadata.json` — dataset and preprocessing metadata
- `rnn_baseline.pt` — PyTorch checkpoint for the baseline model
- `done.txt` — completion marker

Plotting scripts are shared across papers under `papers/shared/time_series/`:

- `python -m papers.shared.time_series.plot_predictions <run_dir>`
- `python -m papers.shared.time_series.plot_metrics <run_dir>`

Training on the Airline Passengers dataset: use `configs/airline_default_rnn.json` (baseline) or `configs/airline_strong_rnn.json` (stronger). Canonical path is `data/time_series/airline-passengers.csv` (legacy `data/QLSTM/airline-passengers.csv` still works).

### Shared time-series plotting

The generic scripts live in `papers/shared/time_series/` and can be used across papers.

## Configuration

Key files:

- `configs/defaults.json` — baseline hyperparameters and dataset paths
- `configs/damped_shm_default_rnn.json` — synthetic dataset (damped SHM) via shared time-series generators
- `cli.json` — CLI schema consumed by the shared runner

Precision control: include `"dtype"` (e.g., `"float32"`) at the top level or under `model` to run in a specific torch dtype.

## Results and Next Steps

- Baseline metrics: mean squared error on validation splits of the weather sequences (see `metrics.json`).
- Planned extensions: implement the QRNN cell described in the paper, run ablations versus the classical RNN, and add visualization notebooks for sequence reconstruction.

## Testing

Run tests from inside `papers/QRNN/`:

```bash
cd papers/QRNN
pytest -q
```

Tests cover the CLI, config loading, and a smoke run of the training loop on the synthetic dataset.
