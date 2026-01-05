import os

import numpy as np
import torch

import photonic_QCNN.data.paper as paper
import photonic_QCNN.data.scratch as scratch


def get_dataset(dataset_name, source, random_state):
    """
    Load and return train/test datasets for the specified dataset.

    Args:
        dataset_name (str): Name of dataset to load. Options: 'BAS', 'Custom BAS', 'MNIST'
        source (str): Data source. Options: 'paper' or 'scratch'
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (x_train, x_test, y_train, y_test) arrays

    Raises:
        AssertionError: If dataset_name or source are invalid
    """

    assert dataset_name in ["BAS", "Custom BAS", "MNIST"], (
        f"Invalid dataset name: {dataset_name}"
    )
    assert source in ["paper", "scratch"], f"Invalid source: {source}"

    if dataset_name == "BAS":
        if source == "paper":
            x_train, x_test, y_train, y_test = paper.get_bas()
        else:
            x_train, x_test, y_train, y_test = scratch.get_bas(
                random_state=random_state
            )
    elif dataset_name == "Custom BAS":
        if source == "paper":
            x_train, x_test, y_train, y_test = paper.get_custom_bas()
        else:
            x_train, x_test, y_train, y_test = scratch.get_custom_bas(
                random_state=random_state
            )
    else:
        if source == "paper":
            x_train, x_test, y_train, y_test = paper.get_mnist()
        else:
            x_train, x_test, y_train, y_test = scratch.get_mnist(
                random_state=random_state
            )
    return x_train, x_test, y_train, y_test


def get_dataset_description(x_train, x_test, y_train, y_test, dataset_name):
    """
    Generate a formatted string description of dataset statistics.

    Args:
        x_train, x_test: Training and testing input arrays
        y_train, y_test: Training and testing label arrays
        dataset_name (str): Name of the dataset for display purposes

    Returns:
        str: Formatted description with shape, dtype, and statistical information
    """
    arrays = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    s = ""

    for name, array in arrays.items():
        s += f"\n📊{dataset_name} - {name}\n"
        s += "-" * (len(name) + 4) + "\n"
        s += f"Shape      : {array.shape}\n"
        s += f"Dtype      : {array.dtype}\n\n"

        # Flatten for scalar stats
        flat = array.flatten()

        if np.issubdtype(array.dtype, np.number):
            s += f"Min        : {flat.min()}\n"
            s += f"Max        : {flat.max()}\n"
            s += f"Mean       : {flat.mean():.4f}\n"
            s += f"Std        : {flat.std():.4f}\n\n"
        else:
            s += "Non-numeric data\n\n"

        # Unique values
        unique_vals = np.unique(array)
        if unique_vals.size <= 20:  # Don't print a huge list
            s += f"Unique vals: {unique_vals}\n"
        else:
            s += f"Unique vals: {unique_vals[:10]} ... (total {len(unique_vals)})\n"

    return s


def save_dataset_description(x_train, x_test, y_train, y_test, dataset_name, save_path):
    """
    Save dataset description to a file.

    Args:
        x_train, x_test: Training and testing input arrays
        y_train, y_test: Training and testing label arrays
        dataset_name (str): Name of the dataset
        save_path (str): File path where description will be saved
    """
    s = get_dataset_description(x_train, x_test, y_train, y_test, dataset_name)
    os.makedirs("datasets_details", exist_ok=True)
    with open(save_path, "w") as f:
        f.write(s)
    return


def convert_dataset_to_tensor(x_train, x_test, y_train, y_test):
    """
    Convert numpy arrays to PyTorch tensors with appropriate dtypes.

    Args:
        x_train, x_test: Input arrays (converted to float32)
        y_train, y_test: Label arrays (converted to long)

    Returns:
        tuple: Converted tensors (x_train, x_test, y_train, y_test)
    """
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return x_train, x_test, y_train, y_test


def convert_tensor_to_loader(x_train, y_train, batch_size=6):
    """
    Create a PyTorch DataLoader from tensors.

    Args:
        x_train (torch.Tensor): Training input tensor
        y_train (torch.Tensor): Training label tensor
        batch_size (int): Batch size for the DataLoader

    Returns:
        DataLoader: PyTorch DataLoader with shuffling enabled
    """
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader


def convert_scalar_labels_to_onehot(y_train, y_test):
    """
    Convert scalar labels (0,1) to one-hot encoded format.

    Args:
        y_train (np.array): Training labels as scalars
        y_test (np.array): Testing labels as scalars

    Returns:
        tuple: (y_train_onehot, y_test_onehot) as one-hot arrays
    """
    # Number of classes (0 and 1)
    num_classes = 2

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Create one-hot encoding for y_train
    y_train_onehot = np.eye(num_classes)[y_train]

    # Create one-hot encoding for y_test
    y_test_onehot = np.eye(num_classes)[y_test]

    return y_train_onehot, y_test_onehot


if __name__ == "__main__":
    # Collect all datasets from scratch and from paper -> Save their descriptions at ./datasets_details/

    scratch_bas_x_train, scratch_bas_x_test, scratch_bas_y_train, scratch_bas_y_test = (
        get_dataset("BAS", "scratch", 42)
    )
    save_dataset_description(
        scratch_bas_x_train,
        scratch_bas_x_test,
        scratch_bas_y_train,
        scratch_bas_y_test,
        "BAS",
        "./datasets_details/BAS_scratch.txt",
    )

    (
        scratch_c_bas_x_train,
        scratch_c_bas_x_test,
        scratch_c_bas_y_train,
        scratch_c_bas_y_test,
    ) = get_dataset("Custom BAS", "scratch", 42)
    save_dataset_description(
        scratch_c_bas_x_train,
        scratch_c_bas_x_test,
        scratch_c_bas_y_train,
        scratch_c_bas_y_test,
        "Custom BAS",
        "./datasets_details/custom_BAS_scratch.txt",
    )

    (
        scratch_mnist_x_train,
        scratch_mnist_x_test,
        scratch_mnist_y_train,
        scratch_mnist_y_test,
    ) = get_dataset("MNIST", "scratch", 42)
    save_dataset_description(
        scratch_mnist_x_train,
        scratch_mnist_x_test,
        scratch_mnist_y_train,
        scratch_mnist_y_test,
        "MNIST",
        "./datasets_details/MNIST_scratch.txt",
    )

    paper_bas_x_train, paper_bas_x_test, paper_bas_y_train, paper_bas_y_test = (
        get_dataset("BAS", "paper", 42)
    )
    save_dataset_description(
        paper_bas_x_train,
        paper_bas_x_test,
        paper_bas_y_train,
        paper_bas_y_test,
        "BAS",
        "./datasets_details/BAS_paper.txt",
    )

    paper_c_bas_x_train, paper_c_bas_x_test, paper_c_bas_y_train, paper_c_bas_y_test = (
        get_dataset("Custom BAS", "paper", 42)
    )
    save_dataset_description(
        paper_c_bas_x_train,
        paper_c_bas_x_test,
        paper_c_bas_y_train,
        paper_c_bas_y_test,
        "Custom BAS",
        "./datasets_details/custom_BAS_paper.txt",
    )

    paper_mnist_x_train, paper_mnist_x_test, paper_mnist_y_train, paper_mnist_y_test = (
        get_dataset("MNIST", "paper", 42)
    )
    save_dataset_description(
        paper_mnist_x_train,
        paper_mnist_x_test,
        paper_mnist_y_train,
        paper_mnist_y_test,
        "MNIST",
        "./datasets_details/MNIST_paper.txt",
    )
