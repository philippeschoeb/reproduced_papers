from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROJECT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.plotting import plot_dataset_examples_from_payload  # noqa: E402

from runtime_lib import run_from_project  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_dir(previous_run: str | None) -> Path:
    if previous_run:
        return Path(previous_run).expanduser().resolve()
    return run_from_project(PROJECT_DIR, ["--task", "classify"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot classification datasets.")
    parser.add_argument(
        "--previous-run",
        dest="previous_run",
        help="Path to a previous run directory to reuse classification_datasets.json.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.previous_run)
    payload_path = (
        run_dir / "classify" / "visualization_data" / "classification_datasets.json"
    )
    if not payload_path.exists():
        warnings.warn(
            "Missing classification_datasets.json; the provided previous run has to be a `classify` task run.",
            stacklevel=2,
        )
        raise FileNotFoundError(
            f"Missing classification_datasets.json at {payload_path}. the provided previous run has to be a `classify` task run."
        )

    payload = _load_json(payload_path)
    figures_dir = run_dir / "classify" / "figures" / "datasets"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "classification_datasets.png"
    plot_dataset_examples_from_payload(payload, output_path)
    if args.previous_run:
        print(f"Saved dataset plots to {output_path.resolve()}")


if __name__ == "__main__":
    main()
