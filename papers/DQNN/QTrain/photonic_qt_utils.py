import numpy as np
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "TorchMPS"))

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def setup_session():
    session = None
    if session is not None:
        session.start()
    return session


def create_boson_samplers(session):
    from QTrain.boson_sampler import BosonSampler
    import perceval as pcvl

    bs_1 = BosonSampler(m=9, n=4, session=session)
    print(
        f"Boson sampler defined with number of parameters = {bs_1.nb_parameters}, and embedding size = {bs_1.embedding_size}"
    )

    bs_2 = BosonSampler(m=8, n=4, session=session)
    print(
        f"Boson sampler defined with number of parameters = {bs_2.nb_parameters}, and embedding size = {bs_2.embedding_size}"
    )
    return bs_1, bs_2


def calculate_qubits():
    from QTrain.classical_utils import CNNModel

    standard_model = CNNModel(use_weight_sharing=False, shared_rows=10)

    numpy_weights = {}
    nw_list = []
    nw_list_normal = []
    for name, param in standard_model.state_dict().items():
        numpy_weights[name] = param.cpu().numpy()
    for i in numpy_weights:
        nw_list.append(list(numpy_weights[i].flatten()))
    for i in nw_list:
        for j in i:
            nw_list_normal.append(j)
    print("# of NN parameters for quantum circuit: ", len(nw_list_normal))
    n_qubits = int(np.ceil(np.log2(len(nw_list_normal))))
    print("Required qubit number: ", n_qubits)
    n_qubit = n_qubits
    return n_qubit, nw_list_normal


def probs_to_weights(probs_, model_template):
    new_state_dict = {}
    data_iterator = probs_.view(-1)

    for name, param in model_template.state_dict().items():
        shape = param.shape
        num_elements = param.numel()
        chunk = data_iterator[:num_elements].reshape(shape)
        new_state_dict[name] = chunk
        data_iterator = data_iterator[num_elements:]

    return new_state_dict


def generate_qubit_states_torch(n_qubit):
    all_states = torch.cartesian_prod(*[torch.tensor([-1, 1]) for _ in range(n_qubit)])
    return all_states
