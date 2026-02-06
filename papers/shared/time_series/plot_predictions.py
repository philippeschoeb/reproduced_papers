#!/usr/bin/env python3
"""Visualize reference vs predictions for one or more run directories.

Targets the QRNN-style `predictions.csv` (columns: split,prediction,target).

Examples:
  python -m papers.shared.time_series.plot_predictions /tmp/run_2026...
  python -m papers.shared.time_series.plot_predictions /tmp/run_a /tmp/run_b

If you want the QRNN convenience wrapper, use:
  python papers/QRNN/utils/plot_predictions.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_repo_root_on_syspath() -> None:
    if __package__:
        return
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_root_on_syspath()

from papers.shared.time_series.plotting import maybe_use_agg_backend, save_figure


def _load_config(run_dir: Path) -> dict:
    snapshot = run_dir / "config_snapshot.json"
    if snapshot.exists():
        return json.loads(snapshot.read_text())
    return {}


def _build_title(cfg: dict, run_dirs: list[Path]) -> str:
    dataset_name = cfg.get("dataset", {}).get("name", "dataset")
    model_name = cfg.get("model", {}).get("name", "model")
    labels = ", ".join(rd.name for rd in run_dirs)
    return f"Predictions overlay — {dataset_name} ({model_name}) — {labels}"


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions.csv in {run_dir}")
    df = pd.read_csv(predictions_path)
    df["run"] = run_dir.name
    return df


def _order_with_steps(
    df: pd.DataFrame, splits: Iterable[str]
) -> tuple[pd.DataFrame, dict]:
    ordered = [df[df["split"] == split] for split in splits]
    df_ordered = pd.concat(ordered, ignore_index=True)
    df_ordered["step"] = range(len(df_ordered))

    boundaries = {}
    cursor = 0
    for split_df, split in zip(ordered, splits):
        cursor += len(split_df)
        boundaries[split] = cursor
    return df_ordered, boundaries


def _assert_compatibility(base: pd.DataFrame, other: pd.DataFrame) -> None:
    if len(base) != len(other):
        raise ValueError(
            "Runs are incompatible: different number of rows in predictions.csv"
        )
    if not np.allclose(base["target"].values, other["target"].values):
        raise ValueError("Runs are incompatible: targets differ between runs")


def plot_runs(run_dirs: list[Path], out_path: Path | None = None) -> Path:
    run_dirs = [rd.resolve() for rd in run_dirs]
    cfg = _load_config(run_dirs[0])
    y_label = cfg.get("dataset", {}).get("target_column", "target")

    maybe_use_agg_backend(show=False)
    import matplotlib.pyplot as plt

    splits = ["train", "val", "test"]
    dfs = [_load_predictions(rd) for rd in run_dirs]

    base_ordered, boundaries = _order_with_steps(dfs[0], splits)
    ordered_dfs = [base_ordered]
    for df in dfs[1:]:
        ordered, _ = _order_with_steps(df, splits)
        _assert_compatibility(base_ordered, ordered)
        ordered_dfs.append(ordered)

    plt.figure(figsize=(12, 5))

    combined = pd.concat(ordered_dfs, ignore_index=True)
    ymin, ymax = (
        combined[["target", "prediction"]].min().min(),
        combined[["target", "prediction"]].max().max(),
    )
    ypad = 0.05 * (ymax - ymin if ymax != ymin else 1.0)
    ymin -= ypad
    ymax += ypad

    colors = {"train": "#ace5b1", "val": "#e0c69c", "test": "#82bae2"}
    start = 0
    for split in splits:
        end = boundaries.get(split, len(base_ordered))
        plt.axvspan(
            start, end, color=colors.get(split, "#f5f5f5"), alpha=0.4, label=None
        )
        plt.text(
            (start + end) / 2,
            ymax - ypad,
            split,
            ha="center",
            va="top",
            fontsize=10,
            alpha=0.8,
        )
        start = end

    plt.plot(
        base_ordered["step"],
        base_ordered["target"],
        label="reference",
        linewidth=2,
        color="black",
    )
    for ordered in ordered_dfs:
        label = ordered["run"].iloc[0] if "run" in ordered else "prediction"
        plt.plot(ordered["step"], ordered["prediction"], label=label, linewidth=1.5)

    plt.title(_build_title(cfg, run_dirs))
    plt.xlabel("sequence index (train → val → test)")
    plt.ylabel(y_label)
    plt.ylim([ymin, ymax])
    plt.legend()
    plt.tight_layout()

    if out_path is None:
        suffix = (
            "predictions_overlay.png" if len(run_dirs) > 1 else "predictions_plot.png"
        )
        out_path = run_dirs[0] / suffix

    out_path = save_figure(plt, out_path, dpi=150)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot predictions for one or more run directories."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more run directories containing predictions.csv",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Explicit output path for the saved plot (default: saved in the first run directory)",
    )
    args = parser.parse_args(argv)

    out_path = plot_runs(args.run_dirs, out_path=args.out_path)
    print(f"Saved plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
