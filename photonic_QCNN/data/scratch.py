import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from photonic_QCNN.data.bars_and_stripes import generate_bars_and_stripes


def get_bas(random_state):
    """
    Generate Bars and Stripes dataset from scratch with reproducible randomness.

    Creates synthetic bars and stripes patterns using the generate_bars_and_stripes
    function with specified noise level and random seed.

    Args:
        random_state (int): Random seed for reproducible generation

    Returns:
        tuple: (x_train, x_test, y_train, y_test) where:
            - x_train: (400, 4, 4) training images
            - x_test: (200, 4, 4) test images
            - y_train: (400,) training labels (0 for bars, 1 for stripes)
            - y_test: (200,) test labels
    """
    np.random.seed(random_state)
    bas_x_all, bas_y_all = generate_bars_and_stripes(600, 4, 4, 0.5)
    bas_x_all = bas_x_all.squeeze(1)  # Remove channel dimension

    # Split into 400 train and 200 test
    bas_x_train, bas_x_test, bas_y_train, bas_y_test = train_test_split(
        bas_x_all, bas_y_all, test_size=200, train_size=400, random_state=random_state)

    bas_x_train = np.reshape(bas_x_train, (400, 4, 4))
    bas_x_test = np.reshape(bas_x_test, (200, 4, 4))

    # Convert labels from -1, 1 to 0, 1
    bas_y_train = (bas_y_train == 1)
    bas_y_test = (bas_y_test == 1)
    bas_y_train = bas_y_train.astype(float)
    bas_y_test = bas_y_test.astype(float)

    return bas_x_train, bas_x_test, bas_y_train, bas_y_test


def get_custom_bas(random_state):
    """
    Generate Custom Bars and Stripes dataset with Gaussian noise.

    Creates bars and stripes patterns with targeted Gaussian noise applied
    only to high-value pixels (originally 1), making classification harder.

    Args:
        random_state (int): Random seed for reproducible noise generation

    Returns:
        tuple: (x_train, x_test, y_train, y_test) where:
            - x_train: (400, 4, 4) noisy training images
            - x_test: (200, 4, 4) noisy test images
            - y_train: (400,) training labels (0 for bars, 1 for stripes)
            - y_test: (200,) test labels
    """
    custom_bas_x, custom_bas_y = generate_bars_and_stripes(600, 4, 4, 0)

    # Only add Gaussian noise to pixels of high value (=1).
    # That results in images for which all low value pixels will be equal to -1.
    mask = (custom_bas_x == 1)
    rng = np.random.default_rng(seed=random_state)
    noise = rng.normal(0, 0.5, size=custom_bas_x.shape)
    noisy_bas_x = custom_bas_x + noise * mask

    # Change low pixel values from -1 to 0
    # noisy_bas_x[noisy_bas_x == -1] = 0  # Weirdly enough this modification really hurts the performance

    c_bas_x_train, c_bas_x_test, c_bas_y_train, c_bas_y_test = train_test_split(
        noisy_bas_x, custom_bas_y, test_size=200, train_size=400, random_state=random_state)
    c_bas_x_train = np.reshape(c_bas_x_train, (400, 4, 4))
    c_bas_x_test = np.reshape(c_bas_x_test, (200, 4, 4))

    # Convert labels from -1, 1 to 0, 1
    c_bas_y_train = (c_bas_y_train == 1)
    c_bas_y_test = (c_bas_y_test == 1)
    c_bas_y_train = c_bas_y_train.astype(float)
    c_bas_y_test = c_bas_y_test.astype(float)

    return c_bas_x_train, c_bas_x_test, c_bas_y_train, c_bas_y_test


def get_mnist(random_state, class_list=(0, 1)):
    """
    Load and preprocess MNIST digits for binary classification.

    Uses sklearn's load_digits (8x8 downsampled MNIST) and filters to
    only include specified digit classes for binary classification tasks.

    Args:
        random_state (int): Random seed for train/test split
        class_list (tuple): Digit classes to include (default: (0, 1))

    Returns:
        tuple: (x_train, x_test, y_train, y_test) where:
            - x_train: (160, 8, 8) training images
            - x_test: (200, 8, 8) test images
            - y_train: (160,) training labels
            - y_test: (200,) test labels
    """
    mnist_x, mnist_y = load_digits(return_X_y=True)

    # Keep only selected classes
    mask = np.isin(mnist_y, class_list)
    mnist_x = mnist_x[mask]
    mnist_y = mnist_y[mask]

    # Train/test split
    mnist_x_train, mnist_x_test, mnist_y_train, mnist_y_test = train_test_split(
        mnist_x, mnist_y, test_size=200, random_state=random_state)
    # Since there are only 360 data points in this specific dataset with labels = 0 or 1, that implies that we will
    # have 160 training points. The paper implementation does something slightly different so they have 154 training
    # points. We do not consider this difference significant.

    # Reshape to 8×8 images
    mnist_x_train = mnist_x_train.reshape(-1, 8, 8)
    mnist_x_test = mnist_x_test.reshape(-1, 8, 8)

    return mnist_x_train, mnist_x_test, mnist_y_train, mnist_y_test
