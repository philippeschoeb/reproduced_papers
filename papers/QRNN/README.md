# QRNN — Quantum Recurrent Neural Networks (Neural Networks, 2023)

This project bootstraps a reproduction of the paper on quantum recurrent neural networks (QRNN, Neural Networks, 2023). It includes both a classical RNN baseline and an experimental photonic QRNN prototype implemented with MerLin/Perceval.

## Reference and Attribution

- Paper: Quantum Recurrent Neural Networks for sequential learning (Neural Networks, 2023)
- Authors: Yanan Li, Zhimin Wang, Rongbing Han, Shangshang Shi, Jiaxin Li, Ruimin Shang, Haiyong Zheng, Guoqiang Zhong, Yongjian Gu
- DOI/ArXiv: https://doi.org/10.1016/j.neunet.2023.07.003 (publisher page: https://www.sciencedirect.com/science/article/abs/pii/S089360802300360X)
- Original repository (if any): not referenced here
- License and attribution notes: cite the published Neural Networks article when using results derived from this code.

## Overview

This implementation (see [QRNN.md]) provides two model families for comparison:

- a **classical RNN baseline** for sequence forecasting on meteorological time-series data
- an **experimental photonic QRNN prototype** using partial measurement to propagate a hidden photonic register

Note: the paper also discusses a gate-based QRNN variant; this gate-based model is **not** reimplemented in this repository.

Datasets and evaluation focus:

- **Weather dataset (main):** we introduce a practical meteorological forecasting dataset based on the Kaggle Szeged weather data (`budincsevity/szeged-weather`) with a preprocessing step that aggregates daily statistics. The original paper does not specify the exact meteorological dataset instance in a fully reproducible way, so this serves as a close, well-known proxy.
- **Simple generators (sanity checks):** additional smoke testing is provided on lightweight synthetic generators such as `sin` and `damped_shm` (shared under `papers/shared/time_series/`).
- **Airline Passengers (baseline time-series):** we also include configs for the classic Airline Passengers dataset (`airline-passengers.csv`) as an additional reference benchmark.

### Photonic QRNN (experimental)

In addition to the classical baseline, this paper folder now includes an **experimental photonic QRNN prototype** implemented with MerLin/Perceval.

The implementation follows the design decisions we settled on for the QRB/QRNN block:

- **Encoding:** the input at time step `t` (denoted `x_t`) is assumed to have dimension `2*kd` and is mapped **directly** to `2*kd` phase shifters on the D register (dual-rail modes). Concretely, `x_t[i]` sets the phase of parameter `in_i`.
- **Partial measurement:** after the QRB circuit, we apply **partial measurement** on the D register (the first `2*kd` modes). The post-selected computation space is controlled by `measurement_space` (dual-rail by default).
- **Readout:** we keep the full measurement outcome space for the chosen post-selection. Let `p_t` be the probability vector over these outcomes. The model predicts
	`y_t = Linear(p_t)`, i.e. a trainable `torch.nn.Linear(len(p_t), 1)`.
- **Hidden-state propagation:** the next hidden state is derived from the conditional remaining state on the unmeasured modes (H register) and projected back into a dual-rail amplitude vector.
	- The photonic QRNN propagates **all** partial-measurement branches. This yields an exact classical mixture over possible hidden states, but the number of branches can grow exponentially.

Post-selection option for the measured D register (`measurement_space`):

- `measurement_space="dual_rail"` (default): outcomes are restricted to dual-rail bitstrings; output dimension is `2**kd`.
- `measurement_space="unbunched"`: outcomes are any 0/1 occupation patterns with `kd` photons across `2*kd` modes (more general than dual-rail); output dimension becomes `binom(2*kd, kd)`.

Current prototype limitations (intentional, to keep the first version minimal and testable):

- `shots=0` only (statevector mode; no sampling)
- Batched inputs are supported, but the current implementation evaluates samples sequentially because partial measurement returns structured branch objects (expect it to be much slower than the classical baseline).
Branch explosion note: if the measured register has `N_out` possible outcomes, then running `depth=j` layers can create up to `N_out**j` branches at a single time step (and the count also multiplies across time steps). Here `N_out` is `2**kd` in dual-rail, or `binom(2*kd, kd)` in unbunched mode.

To enable the photonic model, set the model block in your config to:

```json
{
	"model": {
		"name": "photonic_qrnn",
		"params": {
			"kd": 3,
			"kh": 1,
			"depth": 2,
			"shots": 0,
			"measurement_space": "dual_rail"
		}
	}
}
```

Important: when `model.name="photonic_qrnn"`, the dataset feature dimension must satisfy `len(dataset.feature_columns) == 2*kd`.

Implementation entry points:

- QRB circuit construction: `lib/qrb_circuit.py`
- Partial-measurement readout utilities: `lib/qrb_readout.py`
- Model: `lib/photonic_qrnn.py`

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

- Datasets are stored under the shared data root: by default `data/` at the repo root. For relative `dataset.path` values, QRNN first looks under `data/QRNN/` and, if the file is not found there, falls back to `data/time_series/` for shared time-series CSVs. You can override the base location with `DATA_DIR=/abs/path` or `python implementation.py --data-root /abs/path ...`.
- The default configuration uses `dataset.path="szeged-weather.csv"` and `dataset.preprocess="szeged_weather"`. If the raw file is missing, it will be downloaded from `budincsevity/szeged-weather`, then preprocessed into `szeged-weather.preprocess.csv` the first time. Subsequent runs reuse the preprocessed file.
- The Szeged-specific preprocessor consumes `Formatted Date`, `Temperature (C)`, `Humidity`, `Wind Speed (km/h)`, and `Pressure (millibars)` and outputs a daily CSV with columns: `date`, `min_temperature`, `max_temperature`, `avg_humidity`, `avg_wind_speed`, `avg_pressure`.
- The output also includes a `month` column (derived from `Formatted Date`) which is used by default as an additional feature.
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

Training on the Airline Passengers dataset: use `configs/airline_default_rnn.json` (baseline) or `configs/airline_strong_rnn.json` (stronger). Canonical path is `data/time_series/airline-passengers.csv`.

### Shared time-series plotting

The generic scripts live in `papers/shared/time_series/` and can be used across papers.

## Configuration

Key files:

- `configs/defaults.json` — baseline hyperparameters and dataset paths
- `configs/damped_shm_default_rnn.json` — synthetic dataset (damped SHM) via shared time-series generators
- `cli.json` — CLI schema consumed by the shared runner

Precision control: include `"dtype"` (e.g., `"float32"`) at the top level or under `model` to run in a specific torch dtype.

### Parameter reference (classical vs photonic)

This section documents the configuration parameters that matter most for the two supported model families:

- **Classical baseline**: `model.name="rnn_baseline"` (PyTorch RNN/GRU/LSTM).
- **Photonic prototype**: `model.name="photonic_qrnn"` (MerLin/Perceval partial measurement).

Common top-level fields

- `description` (str): free-form description required by the shared runtime.
- `dtype` (str): floating point dtype label, e.g. `"float32"` or `"float64"`.

Dataset fields (`dataset.*`)

- `dataset.name` (str): dataset identifier. For synthetic generators, use `"sin"`, `"damped_shm"`, etc.
- `dataset.path` (str|null): CSV filename or path. If null, the generator must be set.
- `dataset.preprocess` (str|null): optional preprocessing key (e.g. `"szeged_weather"`).
- `dataset.generator` (obj|null): enables synthetic generation.
	- `dataset.generator.name` (str): generator key (e.g. `"sin"`).
	- `dataset.generator.params` (obj): generator-specific parameters.
	- `dataset.generator.feature_dim` (int, optional): number of features per time step.
		- For the photonic QRNN, this must be exactly `2*kd`.
		- When the generator produces a 1D signal, the current pipeline places it in feature 0 and zero-pads the remaining features.
- `dataset.sequence_length` (int): number of time steps fed to the model.
- `dataset.prediction_horizon` (int): how far ahead to predict (currently used as a shift in the target).
- `dataset.max_rows` (int|null): optional cap on the number of rows used (applied before splitting).
- `dataset.train_ratio` / `dataset.val_ratio` (float): split ratios; test ratio is implicit.
- `dataset.batch_size` (int): training batch size.
	- Photonic prototype note: batching is supported but evaluated sequentially (performance can be dominated by the quantum simulation).
- `dataset.shuffle` (bool): whether to shuffle samples when building the dataloaders.

Training fields (`training.*`)

- `training.epochs` (int)
- `training.optimizer` (str): currently supports `"adam"`.
- `training.lr` (float)
- `training.weight_decay` (float)
- `training.clip_grad_norm` (float|null): optional gradient norm clipping.
- `training.early_stopping.enabled` (bool)
- `training.early_stopping.patience` (int)
- `training.early_stopping.min_delta` (float)

Classical RNN model fields (`model.name="rnn_baseline"`, `model.params.*`)

- `cell_type` (str): `"rnn"`, `"gru"`, or `"lstm"`.
- `hidden_dim` (int): hidden state size of the RNN.
- `layers` (int): number of stacked recurrent layers.
- `dropout` (float): inter-layer dropout (applied by PyTorch when `layers>1`).
- `input_dropout` (float): dropout on the input features before the recurrent core.
- `bidirectional` (bool): whether to use a bidirectional recurrent core.

Photonic QRNN model fields (`model.name="photonic_qrnn"`, `model.params.*`)

- `kd` (int): number of logical "data" photons (D register). Input feature dimension must be `2*kd`.
- `kh` (int): number of logical "hidden" photons (H register). This is the recurrent memory.
- `depth` (int): number of QRB applications *within a single time step*.
	- Note: this is a prototype/generalization knob and is not a named hyperparameter of the original paper. To stay closest to the paper, set `depth=1`.
- `shots` (int): number of samples.
	- Prototype limitation: must be `0` (statevector mode).
- `measurement_space` (str): post-selection space for the measured D register.
	- `"dual_rail"`: outcome space size `2**kd`.
	- `"unbunched"`: outcome space size `binom(2*kd, kd)`.
	- `"all"` (default): exact branching (can grow exponentially).
	- `"argmax"`: keep only the most likely branch (fast approximation).

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
