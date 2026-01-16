import numpy as np
import pytest
import merlin as ML
from QTrain.boson_sampler import BosonSampler


@pytest.fixture
def setup_session():
    session = None
    if session is not None:
        session.start()
    return session


@pytest.fixture
def bs_1(setup_session):
    return BosonSampler(m=9, n=4, session=setup_session)


@pytest.fixture
def bs_2(setup_session):
    return BosonSampler(m=8, n=4, session=setup_session)


def test_init(bs_1, bs_2):
    # bs_1
    bs = bs_1
    assert bs.nb_parameters == 108
    assert bs.nb_parameters >= bs.num_effective_params
    assert len([i for i in bs.quantum_layer.parameters()])
    assert isinstance(bs.quantum_layer, ML.QuantumLayer)

    # bs_2
    bs = bs_2
    assert bs.nb_parameters == 84
    assert bs.nb_parameters >= bs.num_effective_params
    assert len([i for i in bs.quantum_layer.parameters()])
    assert isinstance(bs.quantum_layer, ML.QuantumLayer)


def test_set_params(bs_1):
    bs = bs_1

    # Setting all params to 0
    bs.set_params(np.zeros(bs_1.num_effective_params))

    bs_params = []

    for i in bs.quantum_layer.parameters():
        bs_params.extend(i.detach().cpu().flatten().tolist())

    np.testing.assert_allclose(
        bs_params, np.zeros(bs_1.num_effective_params), rtol=0.000001
    )

    # Randomly assign parameters 5 times
    for _ in range(5):
        rand_params = np.random.random(bs_1.num_effective_params)
        bs.set_params(rand_params)

        bs_params = []

        for i in bs.quantum_layer.parameters():
            bs_params.extend(i.detach().cpu().flatten().tolist())

        np.testing.assert_allclose(bs_params, rand_params, rtol=0.000001)


def test_embedding_size(bs_1):
    bs = bs_1
    assert bs.embedding_size == 126
    assert len(bs.quantum_layer()) == 126
