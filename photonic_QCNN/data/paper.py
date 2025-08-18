"""
Data fetching heavily inspired by the original implementation at: https://github.com/ptitbroussou/Photonic_Subspace_QML_Toolkit
"""

import os
import pickle
import random

import numpy as np
import pennylane as qml
from sklearn.datasets import load_digits


def get_bas():
    """
    Load Bars and Stripes dataset from PennyLane for paper reproduction.

    Returns:
        tuple: (x_train, x_test, y_train, y_test) where:
            - x_train: (400, 4, 4) training images
            - x_test: (200, 4, 4) test images
            - y_train: (400,) training labels (0 for bars, 1 for stripes)
            - y_test: (200,) test labels
    """
    try:
        # Load BAS dataset from pennylane library
        [ds] = qml.data.load("other", name="bars-and-stripes")
        x_train = np.array(ds.train["4"]["inputs"])  # vector representations images
        y_train = np.array(ds.train["4"]["labels"])  # labels for the above images
        x_test = np.array(ds.test["4"]["inputs"])  # vector representations of images
        y_test = np.array(ds.test["4"]["labels"])  # labels for the above images

        # Reshape the data:
        x_train, x_test = (
            x_train[:400].reshape(400, 4, 4),
            x_test[:200].reshape(200, 4, 4),
        )
        y_train, y_test = y_train[:400], y_test[:200]

        # Transform label -1 to label 0:
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

        return x_train, x_test, y_train, y_test

    except Exception as e:
        print(f"Error loading PennyLane BAS dataset: {e}")
        raise e


def get_custom_bas():
    """
    Load Custom Bars and Stripes dataset from pickled binary file.

    This dataset contains modified bars and stripes patterns with
    additional noise and variations for enhanced difficulty.

    Returns:
        tuple: (x_train, x_test, y_train, y_test) where:
            - x_train: (400, 4, 4) training images
            - x_test: (200, 4, 4) test images
            - y_train: (400,) training labels (0 for bars, 1 for stripes)
            - y_test: (200,) test labels
    """
    file_path = os.path.join(os.path.dirname(__file__), "FULL_DATASET_600_samples.bin")
    with open(file_path, "rb") as f:
        data_downloaded = pickle.load(f)

    random.shuffle(data_downloaded)

    x_train = np.array(
        [data_downloaded[i][1] for i in range(400)]
    )  # vector representations images
    y_train = np.array(
        [data_downloaded[i][0] for i in range(400)]
    )  # labels for the above images
    x_test = np.array(
        [data_downloaded[i][1] for i in range(400, 400 + 200)]
    )  # vector representations of images
    y_test = np.array(
        [data_downloaded[i][0] for i in range(400, 400 + 200)]
    )  # labels for the above images

    # Reshape the data:
    x_train = x_train[:400].reshape(400, 4, 4)
    x_test = x_test[:200].reshape(200, 4, 4)
    y_train = y_train[:400]
    y_test = y_test[:200]

    # Transform label -1 to label 0:
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    return x_train, x_test, y_train, y_test


def get_mnist(class_list=(0, 1)):
    """
    Load MNIST digits dataset for binary classification.

    Uses sklearn's load_digits which provides 8x8 downsampled MNIST.
    Filters to only include specified classes for binary classification.

    Args:
        class_list (tuple): Classes to include (default: (0, 1))

    Returns:
        tuple: (x_train, x_test, y_train, y_test) where:
            - x_train: (154, 8, 8) training images
            - x_test: (200, 8, 8) test images
            - y_train: (154,) training labels
            - y_test: (200,) test labels
    """
    digits = load_digits()
    (x_train, y_train), (x_test, y_test) = (
        (digits.data[:750], digits.target[:750]),
        (digits.data[750:], digits.target[750:]),
    )
    x_train, x_test = (
        x_train.reshape(x_train.shape[0], 8, 8),
        x_test.reshape(x_test.shape[0], 8, 8),
    )

    # Only keep the classes in class_list
    train_list_data_array, train_list_label_array = [], []
    test_list_data_array, test_list_label_array = [], []
    for i in range(x_train.shape[0]):
        if (y_train[i] in class_list) and (len(train_list_data_array) < 500):
            train_list_data_array.append(x_train[i])
            train_list_label_array.append(int(y_train[i]))
    for i in range(x_test.shape[0]):
        if (y_test[i] in class_list) and (len(test_list_data_array) < 200):
            test_list_data_array.append(x_test[i])
            test_list_label_array.append(int(y_test[i]))

    return (
        np.array(train_list_data_array),
        np.array(test_list_data_array),
        np.array(train_list_label_array),
        np.array(test_list_label_array),
    )
