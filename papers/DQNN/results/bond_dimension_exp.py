import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List
import json
from QTrain.photonic_qt_utils import (
    setup_session,
    create_boson_samplers,
    calculate_qubits,
)
from QTrain.model import PhotonicQuantumTrain, train_quantum_model
from QTrain.classical_utils import create_datasets

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def plot_results(
    bond_dimensions: List[int],
    epochs: List[int],
    loss_list_epoch: List[float],
    acc_list_epoch: List[float],
):
    rng = np.random.default_rng(0)

    # ----------------------------
    # Plot styling
    # ----------------------------
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    cmap = plt.cm.magma  # close to the purple->yellow vibe
    colors = [cmap(i) for i in np.linspace(0.15, 0.95, len(bond_dimensions))]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 3.2), sharex=True, gridspec_kw={"wspace": 0.18}
    )

    # ----------------------------
    # (a) Training Loss
    # ----------------------------
    for bd, col in zip(bond_dimensions, colors):
        ax1.plot(epochs, loss_list_epoch, color=col, lw=2, alpha=0.95)
        ax1.fill_between(
            epochs,
            np.array(loss_list_epoch[bd]).mean(axis=0)
            - np.array(loss_list_epoch[bd]).std(axis=0),
            np.array(loss_list_epoch[bd]).mean(axis=0)
            + np.array(loss_list_epoch[bd]).std(axis=0),
            color=col,
            alpha=0.18,
            linewidth=0,
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_xlim(0, epochs[-1])
    ax1.set_ylim(0, 2.2)
    ax1.text(
        -0.12, 1.02, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold"
    )

    # ----------------------------
    # (b) Training Acc
    # ----------------------------
    for bd, col in zip(bond_dimensions, colors):
        ax2.plot(
            epochs,
            acc_list_epoch,
            color=col,
            lw=2,
            alpha=0.95,
            label=f"bond_dim = {bd}",
        )
        ax2.fill_between(
            epochs,
            np.array(acc_list_epoch[bd]).mean(axis=0)
            - np.array(acc_list_epoch[bd]).std(axis=0),
            np.array(acc_list_epoch[bd]).mean(axis=0)
            + np.array(acc_list_epoch[bd]).std(axis=0),
            color=col,
            alpha=0.18,
            linewidth=0,
        )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Acc (%)")
    ax2.set_xlim(0, epochs[-1])
    ax2.set_ylim(20, 100)
    ax2.text(
        -0.12, 1.02, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold"
    )

    # Legend outside the right subplot (like your example)
    leg = ax2.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9
    )

    plt.tight_layout()
    plt.savefig(
        "results_results/bond_dimension_graph.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show()


def main(bond_dimension_to_test: List[int] = np.arange(1, 11)):
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    losses = []
    accuracies = []

    # Bond dimension
    for bd in bond_dimension_to_test:
        session = setup_session()
        bs_1, bs_2 = create_boson_samplers(session)

        train_dataset, val_dataset, train_loader, val_loader, batch_size = (
            create_datasets()
        )

        n_qubit, nw_list_normal = calculate_qubits()

        qt_model = PhotonicQuantumTrain(n_qubit, bond_dim=bd).to(device)

        batch_size_qnn = 1000
        train_loader_qnn = DataLoader(train_dataset, batch_size_qnn, shuffle=True)

        qt_model, qnn_parameters, loss_list_epoch, acc_list_epoch = train_quantum_model(
            qt_model,
            train_loader,
            train_loader_qnn,
            bs_1,
            bs_2,
            n_qubit,
            nw_list_normal,
            num_training_rounds=200,
            num_epochs=5,
        )
        losses.append(loss_list_epoch)
        accuracies.append(acc_list_epoch)

        json_str = json.dumps({"loss_list": losses, "acc_list": accuracies}, indent=4)
        with open(
            current_dir + "/MerLin_exp_results/bond_dimension_data.json", "w"
        ) as f:
            f.write(json_str)

    plot_results(bond_dimension_to_test, np.arange(200), losses, accuracies)


if __name__ == "__main__":
    main()
