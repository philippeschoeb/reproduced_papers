import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import warnings

warnings.filterwarnings("ignore")

from lib.photonic_qt_utils import (
    setup_session,
    create_boson_samplers,
    calculate_qubits,
)
from lib.model import PhotonicQuantumTrain, train_quantum_model, evaluate_model
from lib.classical_utils import create_datasets, train_classical_cnn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def run_default_exp(
    bond_dim: int = 7,
    num_training_rounds: int = 2,
    num_epochs: int = 5,
    classical_epochs: int = 5,
    pruning: bool = False,
    pruning_amount: float = 0.5,
    weight_sharing: bool = False,
    shared_rows: int = 10,
    qu_train_with_cobyla: bool = False,
    num_qnn_train_step: int = 12,
):

    print(f"Running experiment with:")
    print(f"  Pruning: {pruning} (amount: {pruning_amount if pruning else 'N/A'})")
    print(
        f"  Weight sharing: {weight_sharing} (shared rows: {shared_rows if weight_sharing else 'N/A'})"
    )

    session = setup_session()
    bs_1, bs_2 = create_boson_samplers(session)

    train_dataset, val_dataset, train_loader, val_loader, batch_size = create_datasets()

    model = train_classical_cnn(
        train_loader,
        val_loader,
        classical_epochs,
        use_pruning=pruning,
        pruning_amount=pruning_amount,
        use_weight_sharing=weight_sharing,
        shared_rows=shared_rows,
    )

    n_qubit, nw_list_normal = calculate_qubits()

    qt_model = PhotonicQuantumTrain(n_qubit, bond_dim=bond_dim).to(device)

    batch_size_qnn = 1000
    train_loader_qnn = DataLoader(train_dataset, batch_size_qnn, shuffle=True)

    qt_model = train_quantum_model(
        qt_model,
        train_loader,
        train_loader_qnn,
        bs_1,
        bs_2,
        n_qubit,
        nw_list_normal,
        num_training_rounds,
        num_epochs,
        qu_train_with_cobyla=qu_train_with_cobyla,
        num_qnn_train_step=num_qnn_train_step,
    )[0]

    evaluate_model(
        qt_model, train_loader, val_loader, bs_1, bs_2, n_qubit, nw_list_normal
    )
