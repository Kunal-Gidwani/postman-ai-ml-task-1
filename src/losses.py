"""
Loss functions for the manual-backpropagation network (deliverables 1.2, 1.3).

Each loss exposes:

    forward(scores, targets) -> scalar loss
    backward()               -> dL/dscores, same shape as `scores`

`backward()` takes no argument because the loss sits at the top of the graph:
dL/dL is 1, so there is nothing to receive from above.

Both losses reduce with a mean, matching the PyTorch defaults they are checked
against. The 1/N (or 1/(N*D)) factor from that mean shows up in every gradient
below; dropping it is a real bug, not a cosmetic one, because it silently
rescales the effective learning rate by the batch size.
"""

import numpy as np


def softmax(scores):
    r"""
    Row-wise softmax:  p[i, k] = exp(s[i, k]) / sum_j exp(s[i, j])

    Subtracting the row max before exponentiating is the standard stabilisation.
    It changes nothing mathematically, because multiplying numerator and
    denominator by exp(-m) cancels:

        exp(s_k - m) / sum_j exp(s_j - m)
            = [exp(s_k) exp(-m)] / [exp(-m) sum_j exp(s_j)]
            = exp(s_k) / sum_j exp(s_j)

    but it guarantees the largest exponent is exp(0) = 1, so nothing overflows.
    """
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    r"""
    Softmax followed by cross-entropy, fused into one layer.

    Forward, for integer class labels y[i] in {0, ..., K-1}:

        p = softmax(s)
        L = -(1/N) * sum_i log p[i, y[i]]

    Why fuse the two
    ----------------
    Kept separate, the backward pass has to carry the full softmax Jacobian

        dp_k/ds_j =  p_k (1 - p_k)   if j == k        (diagonal)
        dp_k/ds_j = -p_k p_j         if j != k        (off-diagonal)

    Fused, the whole Jacobian collapses. With c the correct class for example i,
    cross-entropy only involves p_c, so dL/dp_c = -1/(N p_c) and dL/dp_k = 0 for
    k != c. Chaining that single non-zero term through the Jacobian:

        j == c :  dL/ds_c = (-1/(N p_c)) * p_c (1 - p_c) = (p_c - 1)/N
        j != c :  dL/ds_j = (-1/(N p_c)) * (-p_c p_j)    =  p_j / N

    Both branches are the same expression once you write the target as a one-hot
    vector y_onehot:

        dL/ds = (p - y_onehot) / N

    The 1/p_c from cross-entropy cancels the p_c from the softmax derivative
    exactly. That cancellation is the whole point: when the network is badly
    wrong, p_c is near zero and the standalone softmax derivative p_c(1 - p_c)
    would crush the gradient to nothing. Fused, the gradient for the correct
    class is (p_c - 1)/N, which is near -1/N — a strong signal precisely when
    the network needs it most.

    Numerical stability
    -------------------
    The loss is computed as a log-sum-exp rather than `log(softmax(...))`:

        -log p[i, c] = -(s[i, c] - m_i - log sum_j exp(s[i, j] - m_i))

    This never evaluates log of a possibly-underflowed probability.
    """

    def __init__(self):
        self._probs = None
        self._targets = None

    def forward(self, scores, targets):
        """
        scores  : (N, K) raw logits, no activation applied yet
        targets : (N,) integer class labels
        """
        targets = np.asarray(targets)
        n = scores.shape[0]

        row_max = scores.max(axis=1, keepdims=True)
        shifted = scores - row_max
        log_denominator = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - log_denominator

        self._probs = np.exp(log_probs)
        self._targets = targets

        correct_log_probs = log_probs[np.arange(n), targets]
        return float(-correct_log_probs.mean())

    def backward(self):
        n = self._probs.shape[0]
        d_scores = self._probs.copy()
        # Subtract the one-hot target: p - y_onehot, built in place by hitting
        # only the correct-class column of each row.
        d_scores[np.arange(n), self._targets] -= 1.0
        return d_scores / n

    def predict(self, scores):
        """Argmax class prediction. Softmax is monotone so argmax(s) works too."""
        return scores.argmax(axis=1)


class MSELoss:
    r"""
    Mean squared error, used for the stretch-goal activation/loss pair.

    Forward, with predictions yhat and targets y both of shape (N, D):

        L = (1/(N*D)) * sum_{i,k} (yhat[i, k] - y[i, k])^2

    The mean is over every element, which is what `torch.nn.MSELoss()` does by
    default, so the harness can compare against it directly.

    Backward
        Only the single term (yhat[i, k] - y[i, k])^2 contains yhat[i, k], so

            dL/dyhat[i, k] = (1/(N*D)) * 2 (yhat[i, k] - y[i, k])

        i.e. dL/dyhat = 2 (yhat - y) / (N*D).

    For the binary stretch demo D = 1, so this reduces to 2(yhat - y)/N.

    Note on pairing with Sigmoid
        Unlike softmax + cross-entropy, sigmoid + MSE does *not* cancel. The
        chain rule gives dL/dz = 2(yhat - y)/(N*D) * yhat(1 - yhat), and that
        yhat(1 - yhat) factor goes to zero whenever the network is confidently
        wrong. This is the saturation weakness of the pair, and it is visible in
        the stretch experiment: it trains, just more slowly than a cancelling
        pair would.
    """

    def __init__(self):
        self._diff = None
        self._size = None

    def forward(self, predictions, targets):
        predictions = np.asarray(predictions, dtype=float)
        targets = np.asarray(targets, dtype=float).reshape(predictions.shape)
        self._diff = predictions - targets
        self._size = self._diff.size
        return float(np.mean(self._diff**2))

    def backward(self):
        return 2.0 * self._diff / self._size
