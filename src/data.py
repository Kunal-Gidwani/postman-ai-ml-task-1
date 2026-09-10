# Using scikit-learn dataset
import numpy as np
from sklearn.datasets import load_digits

def load_digits_split(test_fraction=0.2, seed=0):
    # pixels are 0-16, divide so inputs are 0-1
    dataset = load_digits()
    x = dataset.data.astype(np.float64) / 16.0
    y = dataset.target.astype(np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]

    n_test = int(len(x) * test_fraction)
    return x[n_test:], y[n_test:], x[:n_test], y[:n_test]


def make_binary_parity_split(test_fraction=0.2, seed=0):
    # relabel digits as odd/even for the stretch goal (sigmoid+mse needs a yes/no task)
    x_train, y_train, x_test, y_test = load_digits_split(test_fraction, seed)
    return (
        x_train,
        (y_train % 2).astype(np.float64).reshape(-1, 1),
        x_test,
        (y_test % 2).astype(np.float64).reshape(-1, 1),
    )

def iterate_minibatches(x, y, batch_size, rng):
    # shuffle each epoch so the net doesn't memorize the order
    order = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        index = order[start : start + batch_size]
        yield x[index], y[index]
