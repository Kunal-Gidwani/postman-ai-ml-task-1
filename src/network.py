"""
The feedforward network itself (deliverables 1.1, 1.2, 1.3).

`MLP` is a thin sequential container. All the calculus lives in the individual
layers and in the loss; this class only guarantees the two things that make
backpropagation work:

  * forward runs the layers in order, each one caching what its own backward
    pass will need;
  * backward runs them in exactly the reverse order, threading dL/d(output)
    from one layer into the next.

There is no autograd here and no computation graph being built. The ordering of
the two loops *is* the chain rule.
"""

import numpy as np

from .layers import Linear, ReLU


class MLP:
    r"""
    A stack of layers plus a loss.

    Example: the 64 -> 32 -> 10 classifier used for the digits experiment is

        Linear(64, 32) -> ReLU -> Linear(32, 10) -> SoftmaxCrossEntropy

    Forward
        x -> z1 = x W1 + b1 -> a1 = relu(z1) -> z2 = a1 W2 + b2 -> L

    Backward
        Start at the loss, which produces dL/dz2 directly, then walk back:

            dL/dz2  (from the fused softmax + cross-entropy)
            dL/dW2 = a1^T dL/dz2,   dL/db2 = sum_i dL/dz2
            dL/da1 = dL/dz2 W2^T
            dL/dz1 = dL/da1 * 1[z1 > 0]
            dL/dW1 = x^T dL/dz1,    dL/db1 = sum_i dL/dz1

        Each line is one `layer.backward(...)` call, in reverse order.
    """

    def __init__(self, layers, loss):
        self.layers = list(layers)
        self.loss = loss
        self._last_scores = None

    @classmethod
    def build_classifier(cls, d_in, d_hidden, d_out, loss, rng=None):
        """Convenience constructor for the Linear -> ReLU -> Linear stack."""
        return cls(
            layers=[
                Linear(d_in, d_hidden, rng=rng),
                ReLU(),
                Linear(d_hidden, d_out, rng=rng),
            ],
            loss=loss,
        )

    def forward(self, x):
        """Run the layers in order and return the raw output (logits/predictions)."""
        out = np.asarray(x, dtype=float)
        for layer in self.layers:
            out = layer.forward(out)
        self._last_scores = out
        return out

    def compute_loss(self, x, targets):
        """Forward pass plus loss. Returns (loss, network output)."""
        scores = self.forward(x)
        return self.loss.forward(scores, targets), scores

    def backward(self):
        """
        Populate every layer's gradients.

        Must be called after `compute_loss`, because the loss holds the cached
        probabilities (or residuals) that seed the backward pass.
        """
        d_out = self.loss.backward()
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    # ---- parameter access, used by the optimisers and the gradient check ----

    def parameters(self):
        """
        Flat list of (label, param_array, grad_array) triples.

        The arrays are the live objects, not copies, so an optimiser can update
        them in place and the gradient check can perturb them directly.
        """
        out = []
        for index, layer in enumerate(self.layers):
            grads = dict(layer.grads)
            for name, array in layer.params:
                out.append((f"layer{index}.{name}", array, grads[name]))
        return out

    def zero_grad(self):
        """Reset gradients to zero. Not strictly needed here because every
        backward pass overwrites them, but it makes an accumulation bug loud
        rather than silent if the layers are ever changed to `+=`."""
        for layer in self.layers:
            for _, grad in layer.grads:
                grad.fill(0.0)

    def predict(self, x):
        """Class predictions for a classifier loss."""
        return self.loss.predict(self.forward(x))

    def accuracy(self, x, targets):
        return float((self.predict(x) == np.asarray(targets)).mean())
