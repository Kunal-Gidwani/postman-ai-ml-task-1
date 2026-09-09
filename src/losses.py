"""
forward(scores, targets) -> loss
backward() -> dL/dscores
"""

import numpy as np


def softmax(scores):
    # subtract row max before exp so the largest exponent is e^0 = 1, no overflow
    # doesn't change the output mathematically
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """
    softmax + cross-entropy fused.

    p = softmax(s)
    L = -(1/N) * sum_i log p[i, y[i]]

    fused gradient (derived in writeup):
        dL/ds = (p - y_onehot) / N

    the 1/p_c from CE cancels the p_c from softmax, so the gradient
    is (p_c - 1)/N even when the network is very wrong.
    """
    def __init__(self):
        self._probs = None
        self._targets = None

    def forward(self, scores, targets):
        # scores: (N, K) raw logits    targets: (N,) class labels
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
        # p - onehot: subtract 1 from the correct class column
        d_scores[np.arange(n), self._targets] -= 1.0
        return d_scores / n

    def predict(self, scores):
        # argmax on logits works because softmax is monotone
        return scores.argmax(axis=1)


class MSELoss:
    """
    L = mean((yhat - y)^2)
    dL/dyhat = 2(yhat - y) / (N*D)

    with sigmoid: the chain rule adds yhat(1-yhat), which goes to 0 when
    the network is confidently wrong. that's why the stretch run needs more epochs.
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
