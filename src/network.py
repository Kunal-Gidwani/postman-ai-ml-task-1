"""
MLP container. forward runs layers in order, backward runs them in reverse.
"""

import numpy as np

from .layers import Linear, ReLU


class MLP:
    """
    64 -> 32 -> 10 for digits:
    forward:  x -> z1=xW1+b1 -> relu -> z2=a1W2+b2 -> loss
    backward: loss gives dL/dz2, then each layer walks it back
    """

    def __init__(self, layers, loss):
        self.layers = list(layers)
        self.loss = loss
        self._last_scores = None

    @classmethod
    def build_classifier(cls, d_in, d_hidden, d_out, loss, rng=None):
        return cls(
            layers=[
                Linear(d_in, d_hidden, rng=rng),
                ReLU(),
                Linear(d_hidden, d_out, rng=rng),
            ],
            loss=loss,
        )

    def forward(self, x):
        # runs input through every layer in order
        out = np.asarray(x, dtype=float)
        for layer in self.layers:
            out = layer.forward(out)
        self._last_scores = out
        return out

    def compute_loss(self, x, targets):
        # forward + loss in one call
        scores = self.forward(x)
        return self.loss.forward(scores, targets), scores

    def backward(self):
        # must run after compute_loss (loss caches the probs)
        d_out = self.loss.backward()
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out


    def parameters(self):
        # returns live arrays, not copies, so optimiser can update them in place
        out = []
        for index, layer in enumerate(self.layers):
            grads = dict(layer.grads)
            for name, array in layer.params:
                out.append((f"layer{index}.{name}", array, grads[name]))
        return out

    def zero_grad(self):
        # not strictly needed but makes accumulation bugs loud
        for layer in self.layers:
            for _, grad in layer.grads:
                grad.fill(0.0)

    def predict(self, x):
        return self.loss.predict(self.forward(x))

    def accuracy(self, x, targets):
        return float((self.predict(x) == np.asarray(targets)).mean())
