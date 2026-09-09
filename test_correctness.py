"""
runs 19 checks on the hand-derived gradients, prints PASS/FAIL, exits non-zero on failure.
    1. analytic gradients vs numerical (central difference)
    2. analytic gradients + loss vs torch.autograd
    3. relu/sigmoid/tanh backward vs torch
    4. sigmoid + mse vs torch
    5. gradient shapes match parameter shapes
    6. loss actually goes down after 50 sgd steps
"""

import sys
import numpy as np

from gradient_check import (
    build_test_fixture,
    check_activation_against_torch,
    check_against_torch,
    check_model_gradients,
    check_mse_sigmoid_against_torch,
)

# pass if EITHER relative OR absolute is within tolerance (same rule as numpy.allclose)
# numerical is looser: relu kink near zero can flip the mask and inflate the error (~9e-7 observed)
# torch is tighter: both sides are float64 analytic, should agree near machine precision (~4e-14)
NUMERICAL_RELATIVE_TOLERANCE = 1e-5
NUMERICAL_ABSOLUTE_TOLERANCE = 1e-9
TORCH_RELATIVE_TOLERANCE = 1e-10
TORCH_ABSOLUTE_TOLERANCE = 1e-14

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = []


def check(name, condition, detail=""):
    """print PASS/FAIL and record result"""
    status = PASS if condition else FAIL
    message = f"[{status}] {name}"
    if detail:
        message += f"  ->  {detail}"
    print(message)
    results.append(bool(condition))


def skip(name, reason):
    # printed but doesn't count toward pass/fail
    print(f"[{SKIP}] {name}  ->  {reason}")


def check_gradient(name, relative, absolute, rtol, atol):
    # passes if either relative or absolute is within tolerance
    passed = relative < rtol or absolute < atol
    check(name, passed, f"relative {relative:.3e}, absolute {absolute:.3e}")


def torch_available():
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def section(title):
    print(f"\n{title}")
    print("-" * len(title))


# Numerical gradient check

def test_numerical_gradients():
    section("1. Analytic gradients vs central-difference numerical gradients")

    model, x, targets = build_test_fixture()
    for label, relative, absolute in check_model_gradients(model, x, targets):
        check_gradient(
            f"d(loss)/d({label}) matches numerical",
            relative,
            absolute,
            NUMERICAL_RELATIVE_TOLERANCE,
            NUMERICAL_ABSOLUTE_TOLERANCE,
        )


# torch.autograd comparison

def test_against_torch():
    section("2. Analytic gradients vs torch.autograd")

    if not torch_available():
        skip("torch.autograd comparison", "torch is not installed")
        return

    model, x, targets = build_test_fixture()
    loss_difference, errors = check_against_torch(model, x, targets)

    check(
        "loss matches torch.nn.functional.cross_entropy",
        loss_difference < 1e-12,
        f"absolute difference {loss_difference:.3e}",
    )
    for label, relative, absolute in errors:
        check_gradient(
            f"d(loss)/d({label}) matches torch",
            relative,
            absolute,
            TORCH_RELATIVE_TOLERANCE,
            TORCH_ABSOLUTE_TOLERANCE,
        )


# Activation derivatives in isolation

def test_activations():
    section("3. Activation backward passes vs torch")

    if not torch_available():
        skip("activation derivative checks", "torch is not installed")
        return

    rng = np.random.default_rng(1)
    # include +-40 to hit the sigmoid overflow branch
    z = np.concatenate(
        [rng.normal(size=(6, 5)) * 3.0, np.array([[-40.0, -1.0, 0.0, 1.0, 40.0]])]
    )

    for name in ("relu", "sigmoid", "tanh"):
        relative, absolute = check_activation_against_torch(name, z)
        check_gradient(
            f"{name} backward matches torch",
            relative,
            absolute,
            TORCH_RELATIVE_TOLERANCE,
            TORCH_ABSOLUTE_TOLERANCE,
        )


# Stretch-goal pair

def test_stretch_pair():
    section("4. Stretch pair (Sigmoid + MSE) vs torch")

    if not torch_available():
        skip("sigmoid + MSE check", "torch is not installed")
        return

    rng = np.random.default_rng(2)
    z = rng.normal(size=(12, 1)) * 2.0
    y = rng.integers(0, 2, size=(12, 1)).astype(float)

    loss_difference, relative, absolute = check_mse_sigmoid_against_torch(z, y)
    check(
        "MSE loss matches torch.nn.functional.mse_loss",
        loss_difference < 1e-12,
        f"absolute difference {loss_difference:.3e}",
    )
    check_gradient(
        "d(MSE)/dz through sigmoid matches torch",
        relative,
        absolute,
        TORCH_RELATIVE_TOLERANCE,
        TORCH_ABSOLUTE_TOLERANCE,
    )


# Shape invariant

def test_gradient_shapes():
    section("5. Shape invariant: every gradient matches its parameter")

    model, x, targets = build_test_fixture()
    model.compute_loss(x, targets)
    model.backward()

    for label, param, grad in model.parameters():
        check(
            f"{label} gradient shape",
            param.shape == grad.shape,
            f"parameter {param.shape} == gradient {grad.shape}",
        )


# The loss actually decreases
def test_loss_decreases():
    section("6. Training sanity: loss decreases on a fixed batch")

    from src.optimizers import SGD

    model, x, targets = build_test_fixture(n=64)
    optimizer = SGD(model.parameters(), lr=0.5)

    first_loss, _ = model.compute_loss(x, targets)
    for _ in range(50):
        model.compute_loss(x, targets)
        model.backward()
        optimizer.step()
    final_loss, _ = model.compute_loss(x, targets)

    check(
        "loss decreases over 50 SGD steps",
        final_loss < first_loss,
        f"{first_loss:.4f} -> {final_loss:.4f}",
    )


def main():
    print("=" * 68)
    print("Correctness Harness - Task 1: Manual Backpropagation")
    print("=" * 68)

    test_numerical_gradients()
    test_against_torch()
    test_activations()
    test_stretch_pair()
    test_gradient_shapes()
    test_loss_decreases()

    passed = sum(results)
    total = len(results)
    all_passed = passed == total

    print("\n" + "=" * 68)
    verdict = PASS if all_passed else FAIL
    print(f"Result: {verdict}  ({passed}/{total} checks passed)")
    print("=" * 68)

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
