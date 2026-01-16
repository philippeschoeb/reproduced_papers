import numpy as np
import pytest
import torch
from QTrain.photonic_qt_utils import generate_qubit_states_torch, probs_to_weights
from QTrain.classical_utils import CNNModel


def test_generate_qubit_states_torch():
    for i in range(1, 11):
        all_states = generate_qubit_states_torch(i)
        assert len(all_states) == 2**i
        assert len(set(all_states)) == 2**i


def test_probs_to_weights():
    model_template = CNNModel(use_weight_sharing=False, shared_rows=10)
    state_dict = probs_to_weights(torch.tensor(np.random.random(6690)), model_template)
    model_params = dict(model_template.named_parameters())
    for key, value in state_dict.items():
        param_values = model_params[key]
        print(param_values)
        print(value)
        assert len(param_values) == len(value)
