"""
proves the gradient check would actually catch a real bug
breaks each backward pass on purpose and checks the error spikes above the threshold.
"""

import contextlib
import sys

import numpy as np

import src.layers as layers
import src.losses as losses
from gradient_check import build_test_fixture, check_against_torch, check_model_gradients

# correct impl lands ~1e-6. anything above this is a bug, not noise.
DETECTION_THRESHOLD = 1e-3


@contextlib.contextmanager
def patched(owner, name, replacement):
    # swap a method out, put it back even if something throws
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


def worst_errors():
    # largest relative error across all params
    model, x, targets = build_test_fixture()
    numerical = max(rel for _, rel, _ in check_model_gradients(model, x, targets))
    _, torch_errors = check_against_torch(model, x, targets)
    return numerical, max(rel for _, rel, _ in torch_errors)


#  broken backward passes 
def linear_backward_no_bias_sum(self, d_out):
    # bug: use first row instead of summing over batch
    # shape comes out right either way so this is easy to miss
    self.dW[...] = self._x.T @ d_out
    self.db[...] = d_out[0]
    return d_out @ self.W.T


def linear_backward_untransposed_dx(self, d_out):
    # bug: forgot the transpose on W
    self.dW[...] = self._x.T @ d_out
    self.db[...] = d_out.sum(axis=0)
    return d_out @ self.W


def relu_backward_inverted_mask(self, d_out):
    # bug: passes gradient where relu blocked it, blocks where it passed
    return d_out * ~self._mask


def softmax_ce_backward_no_mean(self):
    # bug: forgot the 1/N, rescales all grads by batch size
    # looks like a bad learning rate, not a gradient bug
    d_scores = self._probs.copy()
    d_scores[np.arange(d_scores.shape[0]), self._targets] -= 1.0
    return d_scores


def softmax_ce_backward_sign_flip(self):
    # bug: p - onehot sign is flipped, so training climbs the loss
    d_scores = self._probs.copy()
    d_scores[np.arange(d_scores.shape[0]), self._targets] -= 1.0
    return -d_scores / d_scores.shape[0]


SABOTAGES = [
    ("db takes d_out[0] instead of d_out.sum(0)", layers.Linear, "backward", linear_backward_no_bias_sum),
    ("dx drops the transpose on W", layers.Linear, "backward", linear_backward_untransposed_dx),
    ("ReLU mask inverted", layers.ReLU, "backward", relu_backward_inverted_mask),
    ("softmax CE missing the 1/N", losses.SoftmaxCrossEntropy, "backward", softmax_ce_backward_no_mean),
    ("softmax CE sign flipped", losses.SoftmaxCrossEntropy, "backward", softmax_ce_backward_sign_flip),
]


def main():
    print("=" * 72)
    print("Harness sensitivity: every sabotage below must be detected")
    print("=" * 72)

    baseline_numerical, baseline_torch = worst_errors()
    print(
        f"\nBaseline (correct)   numerical {baseline_numerical:.3e}"
        f"   torch {baseline_torch:.3e}"
    )
    print(f"Detection threshold  {DETECTION_THRESHOLD:.0e}\n")

    detected = []
    for description, owner, name, replacement in SABOTAGES:
        with patched(owner, name, replacement):
            try:
                numerical, torch_error = worst_errors()
                caught = max(numerical, torch_error) > DETECTION_THRESHOLD
                detail = f"numerical {numerical:.3e}   torch {torch_error:.3e}"
            except ValueError as exc:
                # shape mismatch also counts as caught
                caught = True
                detail = f"rejected as a shape error ({str(exc).split(',')[0]})"

        status = "detected" if caught else "MISSED"
        print(f"  [{status:>8}] {description:<42} {detail}")
        detected.append(caught)

    restored_numerical, _ = worst_errors()
    baseline_intact = abs(restored_numerical - baseline_numerical) < 1e-15

    print(f"\n  [{'ok' if baseline_intact else 'BROKEN':>8}] baseline restored after patching")

    all_detected = all(detected) and baseline_intact
    print("\n" + "=" * 72)
    print(
        f"Result: {'PASS' if all_detected else 'FAIL'}  "
        f"({sum(detected)}/{len(detected)} sabotages detected)"
    )
    print("=" * 72)
    sys.exit(0 if all_detected else 1)


if __name__ == "__main__":
    main()
