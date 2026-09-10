# Postman AI/ML Recruitment Task — 26 Batch

## Task 1: Neural Network with Manual Backpropagation

Built a small feedforward net using only NumPy. Every gradient is derived and coded by hand — no autograd, no `.backward()`.

`torch` is only used in the test harness to cross-check my gradients. Nothing in `src/` imports it.

```
Linear(64 -> 32) -> ReLU -> Linear(32 -> 10) -> Softmax + Cross-Entropy
```

Dataset: scikit-learn `load_digits` (1797 handwritten digits, 8x8, 10 classes). No download needed.

## Setup

Python 3.9+

```bash
pip install -r requirements.txt
```

## How to run

**Check gradients (deliverable 1.4)**
```bash
python test_correctness.py
```
Runs PASS/FAIL checks against numerical gradients and torch.autograd. Exits non-zero on failure. Torch checks skip if torch isn't installed.

```bash
python gradient_check.py              # raw error numbers
python test_harness_sensitivity.py    # proves the harness catches real bugs
```

**Train (deliverable 1.5)**
```bash
python train.py           # both experiments
python train.py --main    # digits classifier only
python train.py --stretch # odd/even stretch goal only
```
Prints loss and accuracy per epoch. Writes plots to `plots/` when matplotlib is installed.

## Results so far

Gradient checks pass. Worst relative errors:
- numerical: ~8.9e-07
- torch.autograd: ~4.1e-14

Training (40 epochs, same starting weights):
- SGD lr=0.5 → train loss 1.11 → 0.009, test accuracy ~95.8%
- Adam lr=0.01 → train loss 1.19 → 0.003, test accuracy ~97.5%

Stretch (Tanh + Sigmoid/MSE, odd vs even): loss 0.15 → 0.008, accuracy ~99.4%.

Loss curve images and the full write-up are still being added.

## Files

```
src/
  layers.py       # Linear, ReLU, Sigmoid, Tanh + backward
  losses.py       # SoftmaxCrossEntropy, MSELoss
  network.py      # MLP container
  optimizers.py   # SGD, Momentum, Adam
  data.py         # load_digits, scale, split, minibatches
train.py
gradient_check.py
test_correctness.py
test_harness_sensitivity.py
requirements.txt
```

Still to add: `plots/`, `WRITEUP.md`
