import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pathlib
import os
import time
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List
import torch.optim as optim
import json
from QTrain.photonic_qt_utils import (
    setup_session,
    create_boson_samplers,
    calculate_qubits,
    probs_to_weights,
    generate_qubit_states_torch,
)
from QTrain.model import PhotonicQuantumTrain, train_quantum_model
from QTrain.classical_utils import create_datasets
import torch.nn.functional as F
from QTrain.utils import MNIST_partial

sys.path.append(os.path.join(os.path.dirname(__file__), "TorchMPS"))
from QTrain.TorchMPS.torchmps import MPS

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def plot(
    params_qt,
    accuracy_qt,
    params_ablation,
    accuracy_ablation,
):
    # --- Approximate data reconstructed from the image ---
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)

    # photonic QT (purple squares with error bars)
    ax.plot(
        params_qt,
        accuracy_qt,
        fmt="s-",
        linewidth=1.4,
        markersize=6,
        color="#6a1b9a",
        markerfacecolor="#6a1b9a",
        markeredgecolor="#4a148c",
        label="photonic QT",
    )

    # ablation study (pink/red circles)
    ax.plot(
        params_ablation,
        accuracy_ablation,
        "o-",
        linewidth=1.4,
        markersize=6,
        color="#d65a73",
        markerfacecolor="#d65a73",
        markeredgecolor="#b23a4e",
        label="ablation study",
    )

    # Axes / styling to match the figure
    ax.set_xlim(0, 3500)
    ax.set_ylim(0, 100)
    ax.set_xlabel("# Trainable Parameters")
    ax.set_ylabel("Testing Accuracy (%)")

    ax.grid(True, which="major", linestyle=":", linewidth=0.9, alpha=0.6)

    leg = ax.legend(loc="center right", frameon=True)
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig("results_results/ablation_graph.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def create_ablation_class(bond):
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(8, 12, kernel_size=5)
            self.fc1 = nn.Linear(12 * 4 * 4, 20)
            self.fc2 = nn.Linear(20, 10)

        def forward(self, x):
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = x.view(x.size(0), -1)  # [N, 32 * 8 * 8]
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    # dataset from csv file, to use for the challenge
    train_dataset = MNIST_partial(split="train")
    val_dataset = MNIST_partial(split="val")
    batch_size = 1000
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    # Instantiate the model and loss function
    model = CNNModel()

    numpy_weights = {}
    nw_list = []
    nw_list_normal = []
    for name, param in model.state_dict().items():
        numpy_weights[name] = param.cpu().numpy()
    for i in numpy_weights:
        nw_list.append(list(numpy_weights[i].flatten()))
    for i in nw_list:
        for j in i:
            nw_list_normal.append(j)
    n_qubit = int(np.ceil(np.log2(len(nw_list_normal))))

    random_tensor = torch.randn(126 * 70, 1).cuda()

    class AblationModule(nn.Module):
        def __init__(self):
            """ """
            super().__init__()

            self.MappingNetwork = MPS(
                input_dim=n_qubit + 1, output_dim=1, bond_dim=bond
            )

        def forward(self, x, qnn_parameters):
            """ """

            probs_ = random_tensor

            # probs_ = trans_res.to(device)
            probs_ = probs_[: len(nw_list_normal)]
            probs_ = probs_.reshape(len(nw_list_normal), 1)

            # Generate qubit states using PyTorch
            qubit_states_torch = generate_qubit_states_torch(n_qubit)[
                : len(nw_list_normal)
            ]
            qubit_states_torch = qubit_states_torch.to(device)

            # Combine qubit states with probability values using PyTorch
            combined_data_torch = torch.cat((qubit_states_torch, probs_), dim=1)
            combined_data_torch = combined_data_torch.reshape(
                len(nw_list_normal), n_qubit + 1
            )

            prob_val_post_processed = self.MappingNetwork(combined_data_torch)
            prob_val_post_processed = (
                prob_val_post_processed - prob_val_post_processed.mean()
            )

            state_dict = probs_to_weights(prob_val_post_processed)

            ########

            dtype = torch.float32  # Ensure all tensors are of this type

            # Convolution layer 1 parameters
            conv1_weight = state_dict["conv1.weight"].to(device).type(dtype)
            conv1_bias = state_dict["conv1.bias"].to(device).type(dtype)

            # Convolution layer 2 parameters
            conv2_weight = state_dict["conv2.weight"].to(device).type(dtype)
            conv2_bias = state_dict["conv2.bias"].to(device).type(dtype)

            # Fully connected layer 1 parameters
            fc1_weight = state_dict["fc1.weight"].to(device).type(dtype)
            fc1_bias = state_dict["fc1.bias"].to(device).type(dtype)

            # Fully connected layer 2 parameters
            fc2_weight = state_dict["fc2.weight"].to(device).type(dtype)
            fc2_bias = state_dict["fc2.bias"].to(device).type(dtype)

            # Convolution 1
            x = F.conv2d(x, conv1_weight, conv1_bias, stride=1)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

            # Convolution 2
            x = F.conv2d(x, conv2_weight, conv2_bias, stride=1)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

            # Flatten
            x = x.view(x.size(0), -1)

            # Fully connected 1
            x = F.linear(x, fc1_weight, fc1_bias)

            # Fully connected 2
            x = F.linear(x, fc2_weight, fc2_bias)

            return x

    return AblationModule().to(device), train_loader, val_loader


def main(bond_dimension_to_test: List[int] = np.arange(2, 17)):
    current_dir = str(pathlib.Path(__file__).parent.resolve())

    loss_ablation = []
    accuracy_ablation = []
    params_ablation = []
    loss_qt = []
    accuracy_qt = []
    params_qt = []

    num_training_rounds = 200
    num_epochs = 5

    for bond in bond_dimension_to_test:
        ablation_model, train_loader, val_loader = create_ablation_class(bond=bond)

        params_ablation.append(
            sum(p.numel() for p in ablation_model.parameters() if p.requires_grad)
        )

        ### Training setting ########################

        step = 1e-3  # Learning rate
        q_delta = (
            2 * np.pi
        )  # Phases are 2 pi periodic --> we get better expressivity by multiplying the values by 2 pi

        init_qnn_parameters = q_delta * np.random.rand(108 + 84)

        qnn_parameters = init_qnn_parameters
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            ablation_model.parameters(), lr=step
        )  # , weight_decay=1e-5, eps=1e-6)

        global images, labels

        #############################################
        ### Training loop ###########################
        #############################################

        loss_list = []
        loss_list_epoch = []
        acc_list_epoch = []
        for round_ in range(num_training_rounds):
            print("-----------------------")

            acc_list = []
            acc_best = 0
            for epoch in range(num_epochs):
                ablation_model.train()
                train_loss = 0
                for i, (images, labels) in enumerate(train_loader):
                    correct = 0
                    total = 0
                    since_batch = time.time()

                    images, labels = (
                        images.to(device),
                        labels.to(device),
                    )  # Move data to GPU
                    optimizer.zero_grad()
                    # Forward pass
                    outputs = ablation_model(images, qnn_parameters=qnn_parameters)
                    # print("output: ", outputs)
                    labels_one_hot = F.one_hot(labels, num_classes=10).float()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # Compute loss
                    loss = criterion(outputs, labels)

                    loss_list.append(loss.cpu().detach().numpy())
                    acc = 100 * correct / total
                    acc_list.append(acc)
                    train_loss += loss.cpu().detach().numpy()

                    if acc > acc_best:
                        acc_best = acc
                    loss.backward()

                    optimizer.step()
                    if (i + 1) % 20 == 0:
                        print(
                            f"Training round [{round_ + 1}/{num_training_rounds}], Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, batch time: {time.time() - since_batch:.2f}, accuracy:  {(acc):.2f}%"
                        )

                train_loss /= len(train_loader)

            #############################################
            loss_list_epoch.append(loss)
            acc_list_epoch.append(acc)
            loss_ablation.append(loss_list_epoch)
            accuracy_ablation.append(acc_list_epoch)
            ################################################################################################################################
            session = setup_session()
            bs_1, bs_2 = create_boson_samplers(session)

            n_qubit, nw_list_normal = calculate_qubits()

            qt_model = PhotonicQuantumTrain(n_qubit, bond_dim=bond).to(device)

            qt_model, qnn_parameters, loss_list_epoch, acc_list_epoch = (
                train_quantum_model(
                    qt_model,
                    train_loader,
                    train_loader,
                    bs_1,
                    bs_2,
                    n_qubit,
                    nw_list_normal,
                    num_training_rounds=num_training_rounds,
                    num_epochs=num_epochs,
                )
            )
            loss_qt.append(loss_list_epoch)
            accuracy_qt.append(acc_list_epoch)

            num_trainable_params = sum(
                p.numel() for p in qt_model.parameters() if p.requires_grad
            )
            params_qt.append(
                num_trainable_params
                + bs_1.num_effective_params
                + bs_2.num_effective_params
            )

            json_str = json.dumps(
                {
                    "loss_qt": loss_qt,
                    "accuracy_qt": accuracy_qt,
                    "params_qt": params_qt,
                    "loss_ablation": loss_ablation,
                    "accuracy_ablation": accuracy_ablation,
                    "params_ablation": params_ablation,
                },
                indent=4,
            )
            with open(
                current_dir + "results/MerLin_exp_results/ablation_data.json", "w"
            ) as f:
                f.write(json_str)


if __name__ == "__main__":
    main()
