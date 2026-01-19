"""
Command-line runner for DQNN experiment variants.

This module parses CLI arguments and dispatches to the selected experiment
routine (default, bond dimension, or ablation).
"""

import torch
import pathlib
import json
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from papers.DQNN.utils.utils import parse_args

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import warnings

warnings.filterwarnings("ignore")

from papers.DQNN.lib.ablation_exp import run_ablation_exp
from papers.DQNN.lib.bond_dimension_exp import run_bond_dimension_exp
from papers.DQNN.lib.default_exp import run_default_exp

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main():
    """
    Entry point for running the selected experiment.

    Returns
    -------
    None
    """
    args = parse_args()
    if args.config:
        with open(
            str(pathlib.Path(__file__).parent.parent.resolve())
            + "/configs/"
            + args.config,
            "r",
        ) as f:
            config = json.load(f)
        for key, value in config.items():
            normalized_key = key.lstrip("-")
            if not hasattr(args, normalized_key):
                continue
            current_value = getattr(args, normalized_key)
            if isinstance(current_value, bool) and isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes", "y", "on"}
            setattr(args, normalized_key, value)

    print(args)

    if args.exp_to_run == "DEFAULT":
        print("Running the DEFAULT experiment")
        run_default_exp(
            bond_dim=args.bond_dim,
            num_training_rounds=args.num_training_rounds,
            num_epochs=args.num_epochs,
            classical_epochs=args.classical_epochs,
            pruning=args.pruning,
            pruning_amount=args.pruning_amount,
            weight_sharing=args.weight_sharing,
            shared_rows=args.shared_rows,
            qu_train_with_cobyla=args.qu_train_with_cobyla,
            num_qnn_train_step=args.num_qnn_train_step,
            generate_graph=not args.dont_generate_graph,
        )
    elif args.exp_to_run == "BOND":
        print("Running the BOND experiment")
        run_bond_dimension_exp(
            bond_dimensions_to_test=args.bond_dimensions_to_test,
            num_training_rounds=args.num_training_rounds,
            num_epochs=args.num_epochs,
            qu_train_with_cobyla=args.qu_train_with_cobyla,
            num_qnn_train_step=args.num_qnn_train_step,
            generate_graph=not args.dont_generate_graph,
        )
    elif args.exp_to_run == "ABLATION":
        print("Running the ABLATION experiment")
        run_ablation_exp(
            bond_dimensions_to_test=args.bond_dimensions_to_test,
            num_training_rounds=args.num_training_rounds,
            num_epochs=args.num_epochs,
            qu_train_with_cobyla=args.qu_train_with_cobyla,
            num_qnn_train_step=args.num_qnn_train_step,
            generate_graph=not args.dont_generate_graph,
        )
    else:
        raise NameError("No experiment with that name")


if __name__ == "__main__":
    main()
