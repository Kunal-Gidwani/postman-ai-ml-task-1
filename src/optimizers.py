"""
SGD, Momentum, Adam. takes gradients from MLP.parameters() and updates weights in place.
"""

import numpy as np


class SGD:
    """
    p = p - lr * grad
    too high lr overshoots, too low crawls.
    """

    def __init__(self, parameters, lr=0.1):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        for _, param, grad in self.parameters:
            param -= self.lr * grad


class Momentum:
    """
    v = beta*v + (1-beta)*grad
    p = p - lr*v

    keeps a running average of gradients. consistent directions build up,
    noisy ones cancel. helps in narrow valleys where sgd bounces.
    """

    def __init__(self, parameters, lr=0.1, beta=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta = beta
        self.velocity = {label: np.zeros_like(param) for label, param, _ in self.parameters}

    def step(self):
        for label, param, grad in self.parameters:
            v = self.velocity[label]
            v *= self.beta
            v += (1.0 - self.beta) * grad
            param -= self.lr * v


class Adam:
    """
    m = b1*m + (1-b1)*g        = gradient mean
    v = b2*v + (1-b2)*g^2      = gradient variance

    m_hat = m / (1-b1^t)       = bias correction
    v_hat = v / (1-b2^t)

    p = p - lr * m_hat / (sqrt(v_hat) + eps)
    """
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {label: np.zeros_like(param) for label, param, _ in self.parameters}
        self.v = {label: np.zeros_like(param) for label, param, _ in self.parameters}

    def step(self):
        self.t += 1
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t

        for label, param, grad in self.parameters:
            m, v = self.m[label], self.v[label]

            m *= self.beta1
            m += (1.0 - self.beta1) * grad

            v *= self.beta2
            v += (1.0 - self.beta2) * grad**2

            m_hat = m / bias1
            v_hat = v / bias2

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
