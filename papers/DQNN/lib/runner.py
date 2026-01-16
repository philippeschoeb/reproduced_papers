import torch
import numpy as np
import argparse
import sys
import os
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import warnings

warnings.filterwarnings("ignore")

from lib.ablation_exp import run_ablation_exp
from lib.bond_dimension_exp import run_bond_dimension_exp
from lib.default_exp import run_default_exp

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def int_list(arg):
    return list(map(int, arg.split(",")))


def parse_args():
    parser = argparse.ArgumentParser(description="Photonic Quantum Training")
    parser.add_argument(
        "--exp_to_run",
        type=str,
        default="DEFAULT",
        help="Which experiment to run between 'DEFAULT', 'BOND', 'ABLATION'  (default: 'DEFAULT')",
    )
    parser.add_argument(
        "--bond_dim",
        type=int,
        default=7,
        help="ONLY FOR THE DEFAULT EXPERIMENT! Bond dimension for MPS (default: 7)",
    )
    parser.add_argument(
        "--bond_dimensions_to_test",
        type=int_list,
        default=np.arange(1, 11),
        help="ONLY FOR THE ABLATION AND BOND DIMENSION EXPERIMENT! Bond dimension to test for MPS. Each number needs to be seperated by commas. (default: np.arange(1, 11))",
    )
    parser.add_argument(
        "--num_training_rounds",
        type=int,
        default=2,
        help="Number of training rounds for quantum model (default: 2)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs for training (default: 5)",
    )
    parser.add_argument(
        "--num_qnn_train_step",
        type=int,
        default=12,
        help="Number of epochs for training the quantum layer (default: 12). If the COBYLA is to be runned, 1000 is the suggested value",
    )
    parser.add_argument(
        "--classical_epochs",
        type=int,
        default=1,
        help="Number of epochs for classical CNN training (default: 1)",
    )
    parser.add_argument(
        "--qu_train_with_cobyla",
        type=bool,
        default=False,
        help="If True, optimize the quntum layer with Cobyla. If false, use the Adam optimizer, which is much faster (default: False)",
    )

    parser.add_argument(
        "--pruning", action="store_true", help="Enable pruning experiment"
    )
    parser.add_argument(
        "--pruning_amount",
        type=float,
        default=0.5,
        help="Pruning amount (default: 0.5)",
    )
    parser.add_argument(
        "--weight_sharing", action="store_true", help="Enable weight sharing experiment"
    )
    parser.add_argument(
        "--shared_rows",
        type=int,
        default=10,
        help="Number of shared rows for weight sharing (default: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.exp_to_run == "DEFAULT":
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
        )
    elif args.exp_to_run == "BOND":
        run_bond_dimension_exp(
            bond_dimensions_to_test=args.bond_dimensions_to_test,
            num_training_rounds=args.num_training_rounds,
            num_epochs=args.num_epochs,
            qu_train_with_cobyla=args.qu_train_with_cobyla,
            num_qnn_train_step=args.num_qnn_train_step,
        )
    elif args.exp_to_run == "ABLATION":
        run_ablation_exp(
            bond_dimensions_to_test=args.bond_dimensions_to_test,
            num_training_rounds=args.num_training_rounds,
            num_epochs=args.num_epochs,
            qu_train_with_cobyla=args.qu_train_with_cobyla,
            num_qnn_train_step=args.num_qnn_train_step,
        )
    else:
        raise NameError("No experiment with that name")


if __name__ == "__main__":
    main()
