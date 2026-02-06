#!/usr/bin/env python3
"""Plot training curves (test_acc and loss) for time-series runs.

Currently targets the QRNN-style `metrics.json` format:
- top-level `history`: list of per-epoch dicts with keys:
  `epoch`, `train_loss`, `val_loss`, `val_accuracy`, `test_loss`, `test_accuracy`

Examples:
  python -m papers.shared.time_series.plot_metrics /tmp/run_*/
  python -m papers.shared.time_series.plot_metrics /tmp/run_2026... --include-val

If you want the QRNN convenience wrapper, use:
  python papers/QRNN/utils/plot_metrics.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> None:
    if __package__:
        return
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_root_on_syspath()

from papers.shared.time_series.plotting import maybe_use_agg_backend, save_figure


def _resolve_run_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name != "metrics.json":
            raise ValueError(
                f"Expected a run directory or a metrics.json file, got file: {path}"
            )
        return path.parent
    return path


def _load_history(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {run_dir}")

    metrics = json.loads(metrics_path.read_text())
    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Invalid metrics.json (missing non-empty 'history') in {run_dir}")
    return history


def _series(history: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for row in history:
        if key not in row:
            raise KeyError(f"Missing '{key}' in history row keys={list(row.keys())}")
        out.append(float(row[key]))
    return out


def plot_runs(
    run_dirs: list[Path],
    *,
    out_path: Path | None = None,
    show: bool = False,
    include_val: bool = False,
) -> Path:
    run_dirs = [_resolve_run_dir(p) for p in run_dirs]

    maybe_use_agg_backend(show=show)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    for run_dir in run_dirs:
        history = _load_history(run_dir)
        epochs = _series(history, "epoch")

        test_acc = _series(history, "test_accuracy")
        test_loss = _series(history, "test_loss")

        label = run_dir.name

        ax1.plot(epochs, test_acc, label=label, linewidth=1.8)
        ax2.plot(epochs, test_loss, label=label, linewidth=1.8)

        if include_val:
            val_acc = _series(history, "val_accuracy")
            val_loss = _series(history, "val_loss")
            ax1.plot(epochs, val_acc, linestyle="--", linewidth=1.2, alpha=0.8)
            ax2.plot(epochs, val_loss, linestyle="--", linewidth=1.2, alpha=0.8)

    ax1.set_title("Accuracy")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("test_acc (%)")
    ax1.grid(True, alpha=0.25)

    ax2.set_title("Loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("test_loss")
    ax2.grid(True, alpha=0.25)

    title = "Time-series learning curves"
    if include_val:
        title += " (solid=test, dashed=val)"
    plt.suptitle(title)

    ax1.legend(loc="lower right")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path is None:
        suffix = "metrics_overlay.png" if len(run_dirs) > 1 else "metrics_plot.png"
        out_path = run_dirs[0] / suffix

    out_path = save_figure(plt, out_path, dpi=150)

    if show:
        plt.show()

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot time-series training curves from one or more run directories."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more run directories (or metrics.json files)",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Explicit output path for the saved plot (default: saved in the first run directory)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in a window after saving",
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Also plot validation curves (dashed)",
    )

    args = parser.parse_args(argv)

    out_path = plot_runs(
        args.run_dirs,
        out_path=args.out_path,
        show=args.show,
        include_val=args.include_val,
    )
    print(f"Saved plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
