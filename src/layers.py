"""
Manual-backpropagation network 
forward(x) / backward(d). d = dL/d(output). 
rows = examples.
"""

import numpy as np


class Layer:

    def forward(self, x):
        raise NotImplementedError

    def backward(self, d_out):
        raise NotImplementedError

    @property
    def params(self):
        return []

    @property
    def grads(self):
        return []


class Linear(Layer):
    """
    z = xW + b
    x: (N, d_in)   W: (d_in, d_out)   b: (d_out,)   z: (N, d_out)

    z[i,j] = sum_k x[i,k] * W[k,j] + b[j]
    dW[k,j] = sum_i dz[i,j] * x[i,k]
    dW = x.T @ dz
    
    db[j] = sum_i dz[i,j]   
    db = dz.sum(axis=0)

    dx[i,k] = sum_j dz[i,j] * W[k,j]
    dx = dz @ W.T
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
        # in-place so optimiser still points at the same dW/db
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
    """
    a = max(0, z)
    dz = da * (z > 0)     
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
    Sigmoid activation: s = 1 / (1 + exp(-z)) 
    Derivative
        ds/dz = s (1 - s). 

            ds/dz = -(1 + e^-z)^-2 * (-e^-z)
                  = e^-z / (1 + e^-z)^2
                  = [1 / (1 + e^-z)] * [e^-z / (1 + e^-z)]
                  = s * (1 - s)
        because e^-z / (1 + e^-z) = (1 + e^-z - 1) / (1 + e^-z) = 1 - s.
        dz = ds * s * (1 - s)
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
    a = tanh(z)                       
    Derivative
        da/dz = 1 - tanh^2(z) = 1 - a^2. 
        (e^z - e^-z) / (e^z + e^-z):
            da/dz = [(e^z + e^-z)^2 - (e^z - e^-z)^2] / (e^z + e^-z)^2
                  = 1 - tanh^2(z)
        dz = da * (1 - a^2).
    """

    def __init__(self):
        self._out = None

    def forward(self, z):
        self._out = np.tanh(z)
        return self._out

    def backward(self, d_out):
        return d_out * (1.0 - self._out**2)
