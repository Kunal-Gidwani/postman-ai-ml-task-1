# Write-up — Task 1: Neural Network with Manual Backpropagation

**Task:** Task 1 — Neural Network with Manual Backpropagation
**Repository:** https://github.com/Kunal-Gidwani/postman-ai-ml-task-1

---

## 1. Overview

A small feedforward network built on NumPy matrix operations, with every
gradient derived by hand and implemented directly. Nothing in `src/` imports
`torch`; autograd appears only inside the correctness harness, where the task
brief explicitly allows it as a trusted reference.

The network trained for deliverable 1.5 is

```
Linear(64 -> 32) -> ReLU -> Linear(32 -> 10) -> Softmax + Cross-Entropy
```

on scikit-learn's `load_digits` (1797 real handwritten digits, 8×8 greyscale,
10 classes). It needs no download and writes nothing to disk, so no dataset
ends up committed to the repository.

I chose ReLU for the hidden activation because its derivative is a plain 0/1
mask, which keeps the interesting part of the derivation on the linear layer
and the loss rather than on algebra I would only be copying. Softmax with
cross-entropy is the natural output pairing for mutually exclusive classes, and
their combined gradient is clean for a reason worth understanding (section 2.4).

---

## 2. The gradients, derived

Notation: `N` is the batch size, inputs are `(N, d_in)` so one row is one
example, and `dA` is shorthand for `∂L/∂A`. Every gradient has exactly the same
shape as the quantity it differentiates.

### 2.1 Linear layer

Forward: `z = x W + b`, shapes `x (N, d_in)`, `W (d_in, d_out)`, `b (d_out,)`.

Elementwise, for example `i` and output unit `j`:

```
z[i, j] = Σ_k x[i, k] · W[k, j] + b[j]
```

**Weights.** `W[k, j]` enters only `z[:, j]`, once per example, multiplied by
`x[:, k]`. Chain rule, summing over the batch:

```
dW[k, j] = Σ_i dz[i, j] · ∂z[i, j]/∂W[k, j] = Σ_i dz[i, j] · x[i, k]
```

That double loop over `(k, j)` with a sum over `i` is exactly a matrix product:

```
dW = xᵀ dz            (d_in, N) @ (N, d_out) -> (d_in, d_out)
```

**Bias.** `b[j]` is added to `z[i, j]` for every example, and
`∂z[i, j]/∂b[j] = 1`, so the gradient accumulates across the batch:

```
db[j] = Σ_i dz[i, j]        ->   db = dz.sum(axis=0)
```

**Input.** `x[i, k]` feeds every output unit `j` of the same example, so sum
over `j`. Summing over the second index of `W` is a product with `Wᵀ`:

```
dx = dz Wᵀ            (N, d_out) @ (d_out, d_in) -> (N, d_in)
```

### 2.2 ReLU

Forward `a = max(0, z)`, so `da/dz` is 1 where `z > 0` and 0 where `z < 0`. At
exactly `z = 0` the function is not differentiable; the standard convention,
and what PyTorch does, is to use 0, hence a strict `> 0` mask:

```
dz = da ⊙ 1[z > 0]
```

This is elementwise, not a matrix product. The implementation caches `z`, not
`a`: after the forward pass `a` is zero both where `z` was negative and where
`z` happened to be exactly zero, so rebuilding the mask from the output would
be ambiguous.

### 2.3 Softmax

For `p_k = e^{z_k} / S` with `S = Σ_j e^{z_j}`, the quotient rule gives two
cases. Diagonal (`j = k`), where both numerator and denominator depend on `z_k`:

```
∂p_k/∂z_k = (e^{z_k} S - e^{z_k} e^{z_k}) / S² = p_k - p_k² = p_k(1 - p_k)
```

Off-diagonal (`j ≠ k`), where the numerator `e^{z_k}` is constant in `z_j` and
only `S` moves:

```
∂p_k/∂z_j = (0 · S - e^{z_k} e^{z_j}) / S² = -p_k p_j
```

The minus sign is the intuition that raising any one score steals probability
from all the others.

### 2.4 Softmax + cross-entropy, fused

Forward, for integer labels `y[i]`:

```
L = -(1/N) Σ_i log p[i, y[i]]
```

Cross-entropy touches only the correct class `c`, so `dL/dp_c = -1/(N p_c)` and
`dL/dp_k = 0` for `k ≠ c`. Chaining that single non-zero term through the two
Jacobian cases from section 2.3:

```
j = c :   dL/dz_c = (-1/(N p_c)) · p_c(1 - p_c) = (p_c - 1)/N
j ≠ c :   dL/dz_j = (-1/(N p_c)) · (-p_c p_j)   =  p_j/N
```

Written with a one-hot target, both branches are the same expression:

```
dL/dz = (p - y_onehot) / N
```

**Why fusing them matters.** The `1/p_c` from cross-entropy cancels the `p_c`
from the softmax derivative exactly. That cancellation is the whole point.
When the network is badly wrong, `p_c` is near zero, so the standalone softmax
derivative `p_c(1 - p_c)` is also near zero and would crush the incoming
gradient. Fused, the correct-class gradient is `(p_c - 1)/N ≈ -1/N`: a strong
signal precisely when the network needs it most. This is the concrete reason
softmax pairs with cross-entropy rather than with MSE.

### 2.5 Stretch pair: Sigmoid + MSE

Sigmoid `s = 1/(1 + e^{-z})`. Writing `s = (1 + e^{-z})^{-1}`:

```
ds/dz = -(1 + e^{-z})^{-2} · (-e^{-z})
      = e^{-z}/(1 + e^{-z})²
      = [1/(1 + e^{-z})] · [e^{-z}/(1 + e^{-z})]
      = s(1 - s)
```

using `e^{-z}/(1 + e^{-z}) = (1 + e^{-z} - 1)/(1 + e^{-z}) = 1 - s`.

Tanh, by the quotient rule on `(e^z - e^{-z})/(e^z + e^{-z})`:

```
d/dz tanh(z) = 1 - tanh²(z) = 1 - a²
```

MSE over all `N·D` elements, matching `torch.nn.MSELoss()`'s default reduction:

```
L = (1/(N·D)) Σ (ŷ - y)²        dL/dŷ = 2(ŷ - y)/(N·D)
```

**This pair does not cancel.** Composing them gives

```
dL/dz = [2(ŷ - y)/(N·D)] · ŷ(1 - ŷ)
```

and that `ŷ(1 - ŷ)` factor survives. It goes to zero exactly when the network
is confidently wrong, which is the saturation weakness of sigmoid + MSE and the
mirror image of the cancellation in section 2.4. It is visible in the experiment: the
pair trains fine, just needing more epochs

---

## 3. Correctness verification (deliverable 1.4)

`python test_correctness.py` runs 19 checks and exits non-zero on any failure.
All 19 pass.

Two independent references, because each catches what the other cannot:

1. **Central-difference numerical gradients**, `(L(p+ε) - L(p-ε)) / 2ε`. These
   need only the forward pass, so they cannot share a bug with the analytic
   backward pass. This is the check that actually proves the calculus. Central
   rather than forward differences because the first-order Taylor terms cancel,
   leaving `O(ε²)` error instead of `O(ε)`.
2. **`torch.autograd`.** Catches convention mistakes a numerical check is blind
   to — reducing the loss with a sum where PyTorch uses a mean, for instance,
   since the numerical check would happily confirm the gradient of whatever
   loss I actually wrote.

Worst observed error per parameter:

| Gradient   | vs numerical (relative) | vs torch (relative) |
| ---------- | ----------------------- | ------------------- |
| `layer0.W` | 8.918e-07               | 4.122e-14           |
| `layer0.b` | 9.229e-08               | 9.052e-15           |
| `layer2.W` | 1.513e-08               | 2.630e-15           |
| `layer2.b` | 2.173e-10               | 1.215e-16           |

The loss itself matches `torch.nn.functional.cross_entropy` to 4.441e-16.

A passing check only matters if it would fail on a real bug — see [`test_harness_sensitivity.py`](test_harness_sensitivity.py).

---

##  4. Mistakes and how I found them (deliverable 1.6)

###  4.1 Stale gradient references (caught by reasoning, before running)

`Linear.backward` originally did `self.dW = self._x.T @ d_out`. That **rebinds**
`self.dW` to a fresh array. But `MLP.parameters()` hands out references to the
gradient arrays, and the optimiser holds those references across steps — so
after the first backward pass the optimiser would still be pointing at the
original array and would apply the same stale gradient forever, silently.

Every gradient check would have passed, because the checks call `backward()` and
read the gradients immediately. Only training would have been broken, and it
would have looked like a bad learning rate rather than a bug. Fixed by writing
in place with `self.dW[...] = ...`, which keeps array identity stable.

This is the mistake I am least comfortable about, because nothing in the test
suite would have caught it. It came from thinking about object identity while
writing `parameters()`, not from a failing test.

###  4.2 The tanh check failing at 3.2e-10

First full harness run: 18/19 passed, tanh failed at 3.221e-10 against a 1e-10
tolerance. The tempting move was to loosen the tolerance until it passed, which
would have meant never learning whether the gradient was actually right.

Instead I printed the worst offending element: `z = -8.13`, `tanh(z) =
-0.999999827…`, `1 - tanh² = 3.45e-07`, mine `3.1139835583e-07`, torch's
`3.1139835563e-07`, absolute difference **2.006e-16**. Then I checked whether
NumPy's and torch's `tanh` agree at all — they differ by 1.1e-16, one bit.

So the derivation was correct and the metric was wrong: `1 - t²` with
`t ≈ 0.9999998` is catastrophic cancellation, which amplifies a one-bit input
difference into a 3e-10 relative output difference. The real fix was reporting
absolute error alongside relative and passing on either, not moving the
threshold.

###  4.3 Which mistakes the harness would have caught

`test_harness_sensitivity.py` exists because of the near-miss in section  4.1. It
reintroduces the classic errors deliberately and confirms each is detected. The
two I would most likely have made without a check:

- **Omitting the batch sum in `db`.** `b` is shared by all `N` examples, so all
  `N` contributions must be added. This is the easiest gradient in the network
  to get wrong precisely because the shape comes out plausible either way, so
  the shape invariant does not catch it. Relative error 1.000.
- **Losing the `1/N`.** The loss reduces with a mean, so the factor belongs in
  every gradient. It rescales all gradients by the batch size, which again
  presents as a mistuned learning rate rather than a bug. Relative error 0.778.

Shape checking caught the missing transpose in `dx` for free: `dz @ W` cannot
even execute. Making shape-vs-gradient-shape an explicit assertion turned that
from a crash into a named failing check.

###  4.4 Numerical stability, fixed pre-emptively rather than after a NaN

Two places were written defensively from the start because the failure mode is
a silent `NaN` rather than a wrong number:

- Softmax subtracts the row max before exponentiating. This is mathematically a
  no-op (multiplying numerator and denominator by `e^{-m}` cancels) but
  guarantees the largest exponent is `e⁰ = 1`, so nothing overflows.
- Sigmoid is split into branches: `1/(1 + e^{-z})` for `z ≥ 0` and the
  algebraically identical `e^z/(1 + e^z)` for `z < 0`. A single expression
  overflows for large negative `z`. The harness feeds ±40 through the
  activations specifically to exercise this.
