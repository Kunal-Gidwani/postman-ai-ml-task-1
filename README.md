# Postman AI/ML Recruitment Task — 26 Batch

## Task

**Task 1: Neural Network with Manual Backpropagation**

A small feedforward neural network built on NumPy matrix operations, with every
gradient derived by hand and implemented directly. No autograd and no
`.backward()` anywhere in the implementation.

`torch` is used **only** inside the correctness harness, as the trusted
reference that deliverable 1.4 asks for. Nothing in `src/` imports it.

```
Linear(64 -> 32) -> ReLU -> Linear(32 -> 10) -> Softmax + Cross-Entropy
```

trained on scikit-learn's `load_digits` (1797 real handwritten digits, 8x8,
10 classes — no download required, and no dataset committed to the repo).

## Setup

Requires Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

## How to run

### Verify the gradients (deliverable 1.4)

```bash
python test_correctness.py
```

Runs 19 checks and prints PASS/FAIL for each, exiting non-zero on any failure:
analytic gradients against central-difference numerical gradients, the same
against `torch.autograd`, each activation's backward pass in isolation, the
stretch-goal Sigmoid + MSE pair, the shape invariant, and a sanity check that
the loss decreases. The `torch` checks skip rather than fail if `torch` is not
installed.

For the raw error numbers without the pass/fail wrapper:

```bash
python gradient_check.py
```

### Confirm the gradient check is not vacuous

```bash
python test_harness_sensitivity.py
```

Deliberately breaks each backward pass in turn and confirms every bug is
detected. A passing gradient check only means something if it would fail on a
real error, so this measures the margin rather than assuming it.

### Train the network (deliverable 1.5)

```bash
python train.py              # both experiments
python train.py --main       # digits classifier only
python train.py --stretch    # stretch experiment only
```

Prints per-epoch loss and test accuracy, and writes `plots/loss_curve.png` and
`plots/stretch_loss_curve.png`.

## Results

All 19 correctness checks pass. Worst-case gradient agreement:

| Reference | Worst relative error |
|---|---|
| Central-difference numerical gradients | 8.918e-07 |
| `torch.autograd` | 4.122e-14 |

Training, 40 epochs, both optimisers from identical initial weights:

| Optimiser | Train loss (first -> last) | Final test accuracy |
|---|---|---|
| SGD, lr 0.5 | 1.1099 -> 0.0089 | 0.9582 |
| Adam, lr 0.01 | 1.1861 -> 0.0032 | 0.9749 |

Stretch goal (Tanh hidden + Sigmoid/MSE output, odd vs even): MSE 0.1482 ->
0.0078, test accuracy 0.9944.

![Loss curves](plots/loss_curve.png)

## Project structure

```
.
├── src/
│   ├── layers.py       # Linear, ReLU, Sigmoid, Tanh - forward + hand-derived backward
│   ├── losses.py       # SoftmaxCrossEntropy (fused), MSELoss
│   ├── network.py      # MLP container; backward = reversed(layers)
│   ├── optimizers.py   # SGD, Momentum, Adam
│   └── data.py         # load_digits, scaling, split, minibatches
├── train.py                      # training runs, writes the plots
├── gradient_check.py             # numerical + torch.autograd gradient comparison
├── test_correctness.py           # PASS/FAIL harness (deliverable 1.4)
├── test_harness_sensitivity.py   # proves the harness catches real bugs
├── plots/                        # generated loss curves
├── WRITEUP.md                    # derivations, results, mistakes
└── requirements.txt
```

Every gradient is derived in the docstring of the layer that implements it, and
again in [WRITEUP.md](WRITEUP.md) §2.

## Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1.1 | Feedforward network from matrix operations | [`src/network.py`](src/network.py), [`src/layers.py`](src/layers.py) |
| 1.2 | Forward pass: linear layers + activation | `Linear`, `ReLU` in [`src/layers.py`](src/layers.py) |
| 1.3 | Manual backward pass + derivations | `backward()` in each layer; [WRITEUP.md](WRITEUP.md) §2 |
| 1.4 | Gradient check within tolerance | [`test_correctness.py`](test_correctness.py); [WRITEUP.md](WRITEUP.md) §3 |
| 1.5 | Train on real data, loss decreases | [`train.py`](train.py); [WRITEUP.md](WRITEUP.md) §4 |
| 1.6 | Gradient mistakes and how they were fixed | [WRITEUP.md](WRITEUP.md) §5 |
| Stretch | Second activation–loss pair | Tanh + Sigmoid/MSE; [WRITEUP.md](WRITEUP.md) §2.5 |
| Stretch | Optimizer (Momentum / Adam) | [`src/optimizers.py`](src/optimizers.py) |

## Write-up

See [WRITEUP.md](WRITEUP.md) for the full gradient derivations, the reasoning
behind the tolerances, and a discussion of the mistakes made along the way.
