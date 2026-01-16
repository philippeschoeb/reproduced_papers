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
from lib.photonic_qt_utils import (
    setup_session,
    create_boson_samplers,
    calculate_qubits,
    probs_to_weights,
    generate_qubit_states_torch,
)
from lib.model import PhotonicQuantumTrain, train_quantum_model
import torch.nn.functional as F
from lib.utils import MNIST_partial

sys.path.append(os.path.join(os.path.dirname(__file__), "TorchMPS"))
from lib.TorchMPS.torchmps import MPS

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def plot_ablation(
    params_qt,
    accuracy_qt,
    params_ablation,
    accuracy_ablation,
):
    # photonic QT (purple squares with error bars)
    plt.plot(
        params_qt,
        accuracy_qt,
        linewidth=1.4,
        markersize=6,
        color="#6a1b9a",
        markerfacecolor="#6a1b9a",
        markeredgecolor="#4a148c",
        label="photonic QT",
    )

    # ablation study (pink/red circles)
    plt.plot(
        params_ablation,
        accuracy_ablation,
        linewidth=1.4,
        markersize=6,
        color="#d65a73",
        markerfacecolor="#d65a73",
        markeredgecolor="#b23a4e",
        label="ablation study",
    )

    # Axes / styling to match the figure
    plt.xlabel("# Trainable Parameters")
    plt.ylabel("Testing Accuracy (%)")

    plt.grid(True, which="major", linestyle=":", linewidth=0.9, alpha=0.6)

    leg = plt.legend(loc="center right", frameon=True)
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig(
        str(pathlib.Path(__file__).parent.parent.resolve())
        + "/results/ablation_graph.pdf",
        format="pdf",
        bbox_inches="tight",
    )
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

    random_tensor = torch.randn(126 * 70, 1)

    class AblationModule(nn.Module):
        def __init__(self):
            """ """
            super().__init__()

            self.MappingNetwork = MPS(
                input_dim=n_qubit + 1, output_dim=1, bond_dim=bond
            )

        def forward(self, x, qnn_parameters):
            """ """
            from lib.classical_utils import CNNModel

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

            model_template = CNNModel(use_weight_sharing=False, shared_rows=10)
            state_dict = probs_to_weights(prob_val_post_processed, model_template)

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


def run_ablation_exp(
    bond_dimensions_to_test: List[int] = np.arange(2, 17),
    num_training_rounds: int = 2,
    num_epochs: int = 5,
    qu_train_with_cobyla: bool = False,
    num_qnn_train_step: int = 12,
):
    current_dir = str(pathlib.Path(__file__).parent.parent.resolve()) + "/results/"

    loss_ablation = []
    accuracy_ablation = []
    params_ablation = []
    loss_qt = []
    accuracy_qt = []
    params_qt = []

    num_epochs = 5

    for bond in bond_dimensions_to_test:

        ### QTrain
        ## QTrain
        session = setup_session()
        bs_1, bs_2 = create_boson_samplers(session)
        n_qubit, nw_list_normal = calculate_qubits()
        qt_model = PhotonicQuantumTrain(n_qubit, bond_dim=bond).to(device)

        ### Ablation
        ablation_model, train_loader, val_loader = create_ablation_class(bond=bond)

        params_ablation.append(
            sum(p.numel() for p in ablation_model.parameters() if p.requires_grad)
        )

        # Training setting

        step = 1e-3  # Learning rate
        q_delta = 2 * np.pi

        init_qnn_parameters = q_delta * np.random.rand(
            bs_1.num_effective_params + bs_2.num_effective_params
        )

        qnn_parameters = init_qnn_parameters
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            ablation_model.parameters(), lr=step
        )  # , weight_decay=1e-5, eps=1e-6)

        #############################################
        ### Training loop ###########################
        #############################################

        for round_ in range(num_training_rounds):
            print("-----------------------")
            print("Ablation")

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

                    acc = 100 * correct / total
                    acc_list.append(acc)
                    train_loss += loss.cpu().detach().numpy()

                    if acc > acc_best:
                        acc_best = acc
                    loss.backward()

                    optimizer.step()
                    if (i + 1) * 4 % len(train_loader) == 0:
                        print(
                            f"Training round [{round_ + 1}/{num_training_rounds}], Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, batch time: {time.time() - since_batch:.2f}, accuracy:  {(acc):.2f}%"
                        )
                    loss_ab = loss.item()
                    acc_ab = acc

                train_loss /= len(train_loader)

        ################################################################################################################################
        print("QTrain")

        qt_model, qnn_parameters_qt, loss_list_epoch_qt, acc_list_epoch_qt = (
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
                qu_train_with_cobyla=qu_train_with_cobyla,
                num_qnn_train_step=num_qnn_train_step,
            )
        )

        num_trainable_params_qt = sum(
            p.numel() for p in qt_model.parameters() if p.requires_grad
        )

        # Save results
        params_qt.append(
            num_trainable_params_qt
            + bs_1.num_effective_params
            + bs_2.num_effective_params
        )
        loss_qt.append(loss_list_epoch_qt[-1])
        accuracy_qt.append(acc_list_epoch_qt[-1])

        loss_ablation.append(loss_ab)
        accuracy_ablation.append(acc_ab)

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
        with open(current_dir + "ablation_data.json", "w") as f:
            f.write(json_str)

    plot_ablation(
        params_qt=params_qt,
        accuracy_qt=accuracy_qt,
        params_ablation=params_ablation,
        accuracy_ablation=accuracy_ablation,
    )
