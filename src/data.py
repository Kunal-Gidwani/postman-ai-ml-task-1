"""
Dataset loading for deliverable 1.5.

`load_digits` ships inside scikit-learn: 1797 handwritten digits as 8x8
greyscale images, 64 features, 10 classes. It is a real dataset but needs no
download and writes nothing to disk, so the repository stays free of committed
data (the task's git guide explicitly rules out committing datasets).
"""

import numpy as np
from sklearn.datasets import load_digits


def load_digits_split(test_fraction=0.2, seed=0):
    """
    Load the digits dataset, scale it, and split into train/test.

    Pixel values run 0..16, so dividing by 16 puts every feature in [0, 1].
    Unscaled inputs would make the first layer's pre-activations an order of
    magnitude larger, which pushes softmax into its saturated region and slows
    the first few epochs down for no reason.

    The split is a permutation with a fixed seed, so results are reproducible.

    Returns (x_train, y_train, x_test, y_test) with integer class labels.
    """
    dataset = load_digits()
    x = dataset.data.astype(np.float64) / 16.0
    y = dataset.target.astype(np.int64)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]

    n_test = int(len(x) * test_fraction)
    return x[n_test:], y[n_test:], x[:n_test], y[:n_test]


def make_binary_parity_split(test_fraction=0.2, seed=0):
    """
    The same images relabelled as odd (1.0) or even (0.0), for the stretch goal.

    The stretch pair is Sigmoid + MSE, and sigmoid produces one independent
    probability rather than a distribution over classes. Pointing it at the
    10-class problem would be the wrong tool: softmax's outputs compete because
    a digit is exactly one class, whereas sigmoid's do not. Reframing the same
    images as a single yes/no question is the honest way to exercise that pair.

    Returns (x_train, y_train, x_test, y_test) with y of shape (N, 1) in {0, 1}.
    """
    x_train, y_train, x_test, y_test = load_digits_split(test_fraction, seed)
    return (
        x_train,
        (y_train % 2).astype(np.float64).reshape(-1, 1),
        x_test,
        (y_test % 2).astype(np.float64).reshape(-1, 1),
    )


def iterate_minibatches(x, y, batch_size, rng):
    """
    Yield shuffled (x_batch, y_batch) pairs covering one epoch.

    Reshuffling every epoch matters: a fixed order means the network sees the
    same gradient sequence each pass and can settle into a cycle instead of
    converging.
    """
    order = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        index = order[start : start + batch_size]
        yield x[index], y[index]
