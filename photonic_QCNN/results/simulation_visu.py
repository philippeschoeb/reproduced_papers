"""
Usage (after running the MerLin version of the PQCNN):
    python simulation_visu.py
    -> Enter the path to detailed_results.json:

And enter the path to your detailed_results.json file from running the MerLin version of the PQCNN on whichever dataset.

This will create 'simulation_results.png' in the same directory as 'detailed_results.json' which is equivalent to the
Figure 12 from the original paper for the selected dataset.
"""

from matplotlib import pyplot as plt
import json
import numpy as np
import os

def aggregate_loss_per_epoch(loss_history, num_batches):
    return [
        np.mean(loss_history[i*num_batches : (i+1)*num_batches])
        for i in range(len(loss_history) // num_batches)
    ]


def get_loss_acc_visu(detailed_results_path):
    """
    Get from detailed results the simulation visualizations (just like Figure 12 from paper)

    :param detailed_results_path
    """
    output_path = os.path.join(
        os.path.dirname(detailed_results_path),
        "simulation_results.png"
    )

    with open(detailed_results_path, "r") as f:
        data = json.load(f)

    loss_histories = []
    test_loss_histories = []
    train_acc_histories = []
    test_acc_histories = []

    for run_data in data.values():
        losses = run_data["loss_history"]
        test_losses = run_data["test_loss_history"]
        train_accs = run_data["train_acc_history"]
        test_accs = run_data["test_acc_history"]

        n_epochs = len(train_accs)
        num_batches = len(losses) // (n_epochs - 1)

        # Convert batch losses → epoch losses
        epoch_losses = aggregate_loss_per_epoch(losses, num_batches)

        loss_histories.append(epoch_losses)
        test_loss_histories.append(test_losses)
        train_acc_histories.append(train_accs)
        test_acc_histories.append(test_accs)

    # Now all runs have same length
    loss_histories = np.array(loss_histories)
    test_loss_histories = np.array(test_loss_histories)
    train_acc_histories = np.array(train_acc_histories)
    test_acc_histories = np.array(test_acc_histories)

    loss_epochs = np.arange(loss_histories.shape[1])
    accs_epochs = np.arange(train_acc_histories.shape[1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # ---- Loss Figure ----
    ax = axes[1]
    ax.plot(loss_epochs, loss_histories.mean(axis=0), color="blue", label="Mean Train Loss ± Std")
    ax.fill_between(loss_epochs,
                     loss_histories.mean(axis=0) - loss_histories.std(axis=0),
                     loss_histories.mean(axis=0) + loss_histories.std(axis=0),
                     color="blue", alpha=0.2)
    ax.plot(loss_epochs, test_loss_histories.mean(axis=0), color="orange", label="Mean Test Loss ± Std")
    ax.fill_between(loss_epochs,
                    test_loss_histories.mean(axis=0) - test_loss_histories.std(axis=0),
                    test_loss_histories.mean(axis=0) + test_loss_histories.std(axis=0),
                    color="orange", alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    # ---- Accuracy Figure ----
    ax = axes[0]
    ax.plot(accs_epochs, train_acc_histories.mean(axis=0)*100, color="blue", label="Mean Train Acc ± Std")
    ax.fill_between(accs_epochs,
                    (train_acc_histories.mean(axis=0) - train_acc_histories.std(axis=0))*100,
                    (train_acc_histories.mean(axis=0) + train_acc_histories.std(axis=0))*100,
                     color="blue", alpha=0.2)
    ax.plot(accs_epochs, test_acc_histories.mean(axis=0)*100, color="orange", label="Mean Test Acc ± Std")
    ax.fill_between(accs_epochs,
                    (test_acc_histories.mean(axis=0) - test_acc_histories.std(axis=0))*100,
                    (test_acc_histories.mean(axis=0) + test_acc_histories.std(axis=0))*100,
                     color="orange", alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True)

    # ---- Individual Runs Figure ----
    ax = axes[2]
    colors = ['green', 'red', 'blue', 'purple', 'orange']
    for i, (train_acc, test_acc) in enumerate(zip(train_acc_histories, test_acc_histories)):
        ax.plot(accs_epochs, train_acc*100, color=colors[i], linestyle="-", label=f"Train Run {i}")
        ax.plot(accs_epochs, test_acc*100, color=colors[i], linestyle="--", label=f"Test Run {i}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True)

    plt.savefig(output_path)


if __name__ == "__main__":
    detailed_results_path = input("Enter the path to detailed_results.json: ").strip()
    get_loss_acc_visu(detailed_results_path)