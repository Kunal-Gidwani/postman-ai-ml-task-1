"""
Layers for the manual-backpropagation network (deliverables 1.1, 1.2, 1.3).

Every layer exposes the same two methods:

    forward(x)  -> output, caching whatever the backward pass needs
    backward(d) -> gradient with respect to this layer's input

`d` is always dL/d(output of this layer), handed down from the layer above.
Layers that own parameters also fill in their own `.grads` during backward.

Shape convention: inputs are (N, d_in) with N the batch size, so a row is one
example. Every gradient has exactly the same shape as the thing it
differentiates, which is the cheapest sanity check available.
"""

import numpy as np


class Layer:
    """Common interface. Parameterless layers inherit the empty params/grads."""

    def forward(self, x):
        raise NotImplementedError

    def backward(self, d_out):
        raise NotImplementedError

    @property
    def params(self):
        """List of (name, array) for every learnable tensor in this layer."""
        return []

    @property
    def grads(self):
        """List of (name, array) gradients, aligned with `params`."""
        return []


class Linear(Layer):
    r"""
    Fully connected layer: z = x W + b

    Shapes
        x : (N, d_in)
        W : (d_in, d_out)
        b : (d_out,)          broadcast across the batch
        z : (N, d_out)

    Backward pass derivation
    ------------------------
    Write the forward pass elementwise, for example i and output unit j:

        z[i, j] = sum_k x[i, k] * W[k, j] + b[j]

    dL/dW[k, j]
        W[k, j] only enters z[:, j], once per example, multiplied by x[:, k].
        Chain rule and sum over the batch:

            dL/dW[k, j] = sum_i dL/dz[i, j] * dz[i, j]/dW[k, j]
                        = sum_i dz[i, j] * x[i, k]

        That double loop over (k, j) with a sum over i is exactly the matrix
        product x^T @ dz, so

            dW = x^T @ dz                       (d_in, N) @ (N, d_out)

    dL/db[j]
        b[j] is added to z[i, j] for every example i, and dz[i, j]/db[j] = 1,
        so the gradient accumulates over the batch:

            db[j] = sum_i dz[i, j]              -> dz.sum(axis=0)

        Forgetting this sum is the classic bug: b is shared by all N examples,
        so all N contributions must be added, not averaged or taken once.

    dL/dx[i, k]
        x[i, k] feeds every output unit j of the same example, so we sum over j:

            dL/dx[i, k] = sum_j dz[i, j] * W[k, j]

        Summing over the second index of W is a product with W^T:

            dx = dz @ W^T                       (N, d_out) @ (d_out, d_in)

    Initialisation
        He initialisation, W ~ N(0, 2 / d_in), which keeps activation variance
        roughly constant through ReLU layers. Biases start at zero.
    """

    def __init__(self, d_in, d_out, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        self.W = rng.normal(0.0, np.sqrt(2.0 / d_in), size=(d_in, d_out))
        self.b = np.zeros(d_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, d_out):
        # d_out is dL/dz with shape (N, d_out).
        # Written in place so that dW/db keep their identity: the optimiser and
        # the gradient check both hold references to these arrays, and
        # rebinding them here would leave those references pointing at stale
        # gradients from the previous step.
        self.dW[...] = self._x.T @ d_out
        self.db[...] = d_out.sum(axis=0)
        return d_out @ self.W.T

    @property
    def params(self):
        return [("W", self.W), ("b", self.b)]

    @property
    def grads(self):
        return [("W", self.dW), ("b", self.db)]


class ReLU(Layer):
    r"""
    ReLU activation: a = max(0, z)

    Derivative
        da/dz = 1 for z > 0 and 0 for z < 0. At exactly z = 0 the function is
        not differentiable; the standard convention (and what PyTorch does) is
        to use 0, which is why the mask below is a strict `> 0`.

        The upstream gradient is therefore just masked:

            dz = da * 1[z > 0]                  elementwise, not a matmul

    Caching z rather than a keeps the mask honest: after the forward pass a is
    zero both where z was negative and where z happened to be exactly zero, so
    reconstructing the mask from the output would be ambiguous.
    """

    def __init__(self):
        self._mask = None

    def forward(self, z):
        self._mask = z > 0
        return np.where(self._mask, z, 0.0)

    def backward(self, d_out):
        return d_out * self._mask


class Sigmoid(Layer):
    r"""
    Sigmoid activation: s = 1 / (1 + exp(-z))          [stretch goal]

    Derivative
        ds/dz = s (1 - s). Quick derivation, writing s = (1 + e^-z)^-1:

            ds/dz = -(1 + e^-z)^-2 * (-e^-z)
                  = e^-z / (1 + e^-z)^2
                  = [1 / (1 + e^-z)] * [e^-z / (1 + e^-z)]
                  = s * (1 - s)

        because e^-z / (1 + e^-z) = (1 + e^-z - 1) / (1 + e^-z) = 1 - s.

        So the backward pass is dz = ds * s * (1 - s), elementwise.

    The implementation is written piecewise to stay numerically stable: for
    very negative z, exp(-z) overflows, so we use the algebraically identical
    form exp(z) / (1 + exp(z)) on that branch.
    """

    def __init__(self):
        self._out = None

    def forward(self, z):
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        self._out = out
        return out

    def backward(self, d_out):
        return d_out * self._out * (1.0 - self._out)


class Tanh(Layer):
    r"""
    Tanh activation: a = tanh(z)                       [stretch goal]

    Derivative
        da/dz = 1 - tanh^2(z) = 1 - a^2. From the quotient rule on
        (e^z - e^-z) / (e^z + e^-z):

            da/dz = [(e^z + e^-z)^2 - (e^z - e^-z)^2] / (e^z + e^-z)^2
                  = 1 - tanh^2(z)

        so the backward pass is dz = da * (1 - a^2), elementwise.
    """

    def __init__(self):
        self._out = None

    def forward(self, z):
        self._out = np.tanh(z)
        return self._out

    def backward(self, d_out):
        return d_out * (1.0 - self._out**2)
