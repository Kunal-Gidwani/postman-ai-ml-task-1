"""
Gradient verification for deliverable 1.4.

Two independent references, because they fail in different ways:

1. Central-difference numerical gradients. Needs nothing but the forward pass,
   so it cannot share a bug with the analytic backward pass. It is the check
   that actually proves the hand-derived calculus.

2. `torch.autograd`. Catches convention mistakes a numerical check cannot see,
   such as reducing the loss with a sum where PyTorch uses a mean, because the
   numerical check would happily confirm the gradient of whatever loss we
   actually wrote.

`torch` is imported in this file and in `test_correctness.py` only. Nothing in
`src/` imports it, so the implementation stays autograd-free as required.
"""

import numpy as np

from src.layers import Linear, ReLU, Sigmoid, Tanh
from src.losses import MSELoss, SoftmaxCrossEntropy
from src.network import MLP


# ----------------------------------------------------------------------------
# 1. Numerical gradients (central differences)
# ----------------------------------------------------------------------------

def numerical_gradient(loss_fn, param, epsilon=1e-5):
    r"""
    Central-difference estimate of dL/dparam, elementwise.

        dL/dp ~= (L(p + eps) - L(p - eps)) / (2 eps)

    Central differences rather than forward differences because the error is
    O(eps^2) instead of O(eps): the first-order terms of the two Taylor
    expansions cancel. With float64 and eps = 1e-5 that lands around 1e-10
    truncation error against ~1e-11 rounding noise, which is the sweet spot.

    `param` is modified in place and restored, so `loss_fn` must read the live
    array (which it does: it re-runs the forward pass on the same model).
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

        param[index] = original  # restore before moving on
        grad[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

        iterator.iternext()

    return grad


def error_metrics(a, b):
    r"""
    Return (relative_error, absolute_error) between two gradient arrays.

        relative_error = max |a - b| / max(|a| + |b|, tiny)
        absolute_error = max |a - b|

    Both are needed, because each is blind to a case the other catches.

    A plain absolute difference is meaningless without knowing the gradient
    magnitude, so the relative error is the primary metric: it is comparable
    across layers whose gradients differ by orders of magnitude.

    But the relative error becomes misleadingly harsh under catastrophic
    cancellation. The tanh backward pass is the concrete example here. For
    z = -8.13, tanh(z) = -0.999999827..., so evaluating 1 - tanh^2(z) subtracts
    two nearly equal numbers and throws away about seven significant digits.
    NumPy's tanh and PyTorch's tanh disagree in the last bit (~1e-16), and that
    cancellation amplifies the disagreement to ~3e-10 in relative terms even
    though the absolute difference is 2e-16, i.e. machine epsilon. Reporting
    both metrics lets a caller accept that case on the absolute error while
    still holding everything else to a tight relative bound.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    difference = np.abs(a - b)
    denominator = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(difference / denominator)), float(np.max(difference))


def check_model_gradients(model, x, targets, epsilon=1e-5):
    """
    Compare every analytic parameter gradient against central differences.

    Returns a list of (label, relative_error, absolute_error) in the model's
    parameter order.
    """
    model.compute_loss(x, targets)
    model.backward()

    # Snapshot the analytic gradients: the numerical pass perturbs parameters
    # and re-runs forward, which would otherwise overwrite them.
    analytic = [(label, grad.copy()) for label, _, grad in model.parameters()]

    results = []
    for (label, analytic_grad), (_, param, _) in zip(analytic, model.parameters()):
        numeric_grad = numerical_gradient(
            lambda: model.compute_loss(x, targets)[0], param, epsilon
        )
        relative, absolute = error_metrics(analytic_grad, numeric_grad)
        results.append((label, relative, absolute))

    return results


# ----------------------------------------------------------------------------
# 2. torch.autograd reference
# ----------------------------------------------------------------------------

def torch_reference_gradients(model, x, targets):
    """
    Rebuild the exact same 64 -> hidden -> 10 classifier in torch, using our
    weights, and let autograd differentiate it.

    Returns {label: gradient array} keyed the same way as `model.parameters()`.
    """
    import torch

    linear1, _relu, linear2 = model.layers

    x_t = torch.tensor(np.asarray(x, dtype=float), dtype=torch.float64)
    y_t = torch.tensor(np.asarray(targets), dtype=torch.long)

    # torch.nn.Linear stores weight as (out_features, in_features), the
    # transpose of our (d_in, d_out) convention, hence the .T on the way in and
    # again on the way out.
    W1 = torch.tensor(linear1.W.T, dtype=torch.float64, requires_grad=True)
    b1 = torch.tensor(linear1.b, dtype=torch.float64, requires_grad=True)
    W2 = torch.tensor(linear2.W.T, dtype=torch.float64, requires_grad=True)
    b2 = torch.tensor(linear2.b, dtype=torch.float64, requires_grad=True)

    hidden = torch.relu(torch.nn.functional.linear(x_t, W1, b1))
    logits = torch.nn.functional.linear(hidden, W2, b2)

    # CrossEntropyLoss == log_softmax + NLL with a mean reduction, which is
    # exactly what SoftmaxCrossEntropy computes.
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
    """
    Returns (loss_abs_diff, [(label, relative_error, absolute_error), ...]).
    """
    our_loss, _ = model.compute_loss(x, targets)
    model.backward()

    reference = torch_reference_gradients(model, x, targets)

    errors = []
    for label, _, grad in model.parameters():
        relative, absolute = error_metrics(grad, reference[label])
        errors.append((label, relative, absolute))
    return abs(our_loss - reference["loss"]), errors


# ----------------------------------------------------------------------------
# 3. Per-layer checks for the individual activation derivatives
# ----------------------------------------------------------------------------

def check_activation_against_torch(activation_name, z):
    """
    Verify one activation's backward pass in isolation.

    Uses an arbitrary non-uniform upstream gradient rather than ones, so that a
    backward pass which ignores `d_out` (or transposes it) cannot pass by luck.
    """
    import torch

    activations = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}
    torch_fns = {
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
    }

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
    """
    Verify the stretch-goal pair (Sigmoid + MSE) end to end.

    Sigmoid + MSE does not cancel the way softmax + cross-entropy does, so this
    exercises a genuinely different gradient path: dL/dz keeps an explicit
    yhat(1 - yhat) factor.
    """
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


# ----------------------------------------------------------------------------
# Small helper so the harness and the CLI build the same fixture
# ----------------------------------------------------------------------------

def build_test_fixture(n=8, d_in=64, d_hidden=32, d_out=10, seed=0):
    """
    A tiny deterministic model and batch for the checks.

    Deliberately small: the numerical check costs two forward passes per
    parameter, so a 64x32 layer already means ~4k forward passes.
    """
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
