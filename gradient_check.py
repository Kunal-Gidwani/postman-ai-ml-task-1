"""
gradient verification , two references:
1. central-difference numerical gradients - only needs the forward pass, can't share a bug with backward
2. torch.autograd - catches convention mistakes like sum vs mean that numerical check would miss
"""

import numpy as np

from src.layers import Linear, ReLU, Sigmoid, Tanh
from src.losses import MSELoss, SoftmaxCrossEntropy
from src.network import MLP


# numerical gradients

def numerical_gradient(loss_fn, param, epsilon=1e-5):
    """
    (L(p+eps) - L(p-eps)) / (2*eps) for each element.
    """
    grad = np.zeros_like(param)
    iterator = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

    while not iterator.finished:
        index = iterator.multi_index
        original = param[index]

        param[index] = original + epsilon
        loss_plus = loss_fn()

        param[index] = original - epsilon
        loss_minus = loss_fn()

        param[index] = original
        grad[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

        iterator.iternext()

    return grad


def error_metrics(a, b):
    """
    relative = max|a-b| / max(|a|+|b|, tiny)
    absolute = max|a-b|
    need both: relative breaks under catastrophic cancellation (tanh near +-1),
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    difference = np.abs(a - b)
    denominator = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(difference / denominator)), float(np.max(difference))


def check_model_gradients(model, x, targets, epsilon=1e-5):
    # compare every analytic gradient against numerical
    # returns list of (label, relative_error, absolute_error)
    model.compute_loss(x, targets)
    model.backward()

    # snapshot analytic grads before numerical pass overwrites them
    analytic = [(label, grad.copy()) for label, _, grad in model.parameters()]

    results = []
    for (label, analytic_grad), (_, param, _) in zip(analytic, model.parameters()):
        numeric_grad = numerical_gradient(
            lambda: model.compute_loss(x, targets)[0], param, epsilon
        )
        relative, absolute = error_metrics(analytic_grad, numeric_grad)
        results.append((label, relative, absolute))

    return results


# torch.autograd reference

def torch_reference_gradients(model, x, targets):
    # rebuild same model in torch with our weights, let autograd diff it
    import torch

    linear1, _relu, linear2 = model.layers

    x_t = torch.tensor(np.asarray(x, dtype=float), dtype=torch.float64)
    y_t = torch.tensor(np.asarray(targets), dtype=torch.long)

    # torch stores weight as (out, in), we use (in, out), hence the .T
    W1 = torch.tensor(linear1.W.T, dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor(linear1.b, dtype=torch.float64, requires_grad=True)
    W2 = torch.tensor(linear2.W.T, dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor(linear2.b, dtype=torch.float64, requires_grad=True)

    hidden = torch.relu(torch.nn.functional.linear(x_t, W1, b1))
    logits = torch.nn.functional.linear(hidden, W2, b2)

    loss = torch.nn.functional.cross_entropy(logits, y_t, reduction="mean")
    loss.backward()

    return {
        "loss": loss.item(),
        "layer0.W": W1.grad.numpy().T,
        "layer0.b": b1.grad.numpy(),
        "layer2.W": W2.grad.numpy().T,
        "layer2.b": b2.grad.numpy(),
    }


def check_against_torch(model, x, targets):
    # returns (loss_abs_diff, [(label, relative, absolute), ...])
    our_loss, _ = model.compute_loss(x, targets)
    model.backward()

    reference = torch_reference_gradients(model, x, targets)

    errors = []
    for label, _, grad in model.parameters():
        relative, absolute = error_metrics(grad, reference[label])
        errors.append((label, relative, absolute))
    return abs(our_loss - reference["loss"]), errors


# per-activation checks

def check_activation_against_torch(activation_name, z):
    # check one activation's backward in isolation
    # uses random d_out so a wrong backward can't pass by luck
    import torch

    activations = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}
    torch_fns = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}

    layer = activations[activation_name]()
    rng = np.random.default_rng(0)
    d_out = rng.normal(size=z.shape)

    layer.forward(z)
    ours = layer.backward(d_out)

    z_t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    output = torch_fns[activation_name](z_t)
    output.backward(torch.tensor(d_out, dtype=torch.float64))

    return error_metrics(ours, z_t.grad.numpy())


def check_mse_sigmoid_against_torch(z, y):
    # stretch goal: sigmoid + mse end to end
    # doesn't cancel like softmax+ce, so dL/dz keeps the yhat(1-yhat) factor
    import torch

    sigmoid = Sigmoid()
    mse = MSELoss()

    predictions = sigmoid.forward(z)
    our_loss = mse.forward(predictions, y)
    our_grad = sigmoid.backward(mse.backward())

    z_t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    y_t = torch.tensor(np.asarray(y, dtype=float).reshape(z.shape), dtype=torch.float64)
    loss_t = torch.nn.functional.mse_loss(torch.sigmoid(z_t), y_t, reduction="mean")
    loss_t.backward()

    relative, absolute = error_metrics(our_grad, z_t.grad.numpy())
    return abs(our_loss - loss_t.item()), relative, absolute


# shared test fixture

def build_test_fixture(n=8, d_in=64, d_hidden=32, d_out=10, seed=0):
    # small on purpose: numerical check costs 2 forward passes per parameter
    rng = np.random.default_rng(seed)
    model = MLP.build_classifier(d_in, d_hidden, d_out, SoftmaxCrossEntropy(), rng=rng)
    x = rng.normal(size=(n, d_in))
    targets = rng.integers(0, d_out, size=n)
    return model, x, targets


if __name__ == "__main__":
    model, x, targets = build_test_fixture()

    print("Numerical gradient check (central differences)")
    for label, relative, absolute in check_model_gradients(model, x, targets):
        print(f"  {label:<12} relative = {relative:.3e}   absolute = {absolute:.3e}")

    print("\ntorch.autograd comparison")
    loss_diff, errors = check_against_torch(model, x, targets)
    print(f"  {'loss':<12} absolute difference = {loss_diff:.3e}")
    for label, relative, absolute in errors:
        print(f"  {label:<12} relative = {relative:.3e}   absolute = {absolute:.3e}")
