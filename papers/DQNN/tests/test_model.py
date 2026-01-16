import numpy as np
import pytest
import merlin as ML
from QTrain.boson_sampler import BosonSampler
from QTrain.model import PhotonicQuantumTrain, evaluate_model
from QTrain.photonic_qt_utils import calculate_qubits
from QTrain.classical_utils import create_datasets
from tests.test_boson_sampler import bs_1, bs_2, setup_session
import torch.nn as nn
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "TorchMPS"))
from QTrain.TorchMPS.torchmps import MPS


@pytest.fixture
def create_model():
    return PhotonicQuantumTrain(calculate_qubits()[0])


def test_calculate_qubits():
    n_qubits, nw_list = calculate_qubits()
    assert n_qubits == 13
    assert len(nw_list) == 6690


def test_init(create_model):
    model = create_model
    assert isinstance(model, nn.Module)
    assert isinstance(model.MappingNetwork, MPS)
    assert model.MappingNetwork.bond_dim == 7
    assert model.MappingNetwork.input_dim == calculate_qubits()[0] + 1
    assert model.MappingNetwork.output_dim == 1


def test_foward(create_model, bs_1, bs_2):
    model = create_model
    model.train()
    n_qubits, nw_list_norm = calculate_qubits()
    train_dataset = create_datasets()[2]
    index = 0
    for i, (images, _) in enumerate(train_dataset):
        outputs = model(
            images,
            bs_1=bs_1,
            bs_2=bs_2,
            n_qubit=n_qubits,
            nw_list_normal=nw_list_norm,
        )
        _, predicted = torch.max(outputs.data, 1)
        if i < len(train_dataset) - 1:
            assert len(predicted) == 128
        index += 1
        for label in predicted:
            assert label < 10
            assert label >= 0
