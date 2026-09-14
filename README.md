# Postman AI/ML Recruitment Task — 26 Batch

## Task 1: Neural Network with Manual Backpropagation

Built a small feedforward net using NumPy. Every gradient is derived and coded by hand. torch is used in the test harness to cross-check my gradients.

Linear(64 -> 32) -> ReLU -> Linear(32 -> 10) -> Softmax + Cross-Entropy

Dataset: scikit-learn load_digits (1797 handwritten digits, 8x8, 10 classes).

## Setup

Python 3.9+

pip install -r requirements.txt

## How to run

**test run and see prediction**

python predict.py

**Check gradients (deliverable 1.4)**

python test_correctness.py

- Runs 19 checks, prints PASS/FAIL, exits non-zero on failure. Checks analytic vs numerical gradients, vs torch.autograd, each activation separately, and a sanity check that loss goes down.

python gradient_check.py

- just the raw error numbers

**Check the harness isn't useless**

python test_harness_sensitivity.py

- Breaks each backward pass on purpose and confirms every bug gets caught.

**Train (deliverable 1.5)**

python train.py # both experiments
python train.py --main # digits classifier only
python train.py --stretch # odd/even stretch goal only

Prints loss and accuracy per epoch, saves plots to plots/ .

## Results

All 19 checks pass.

| Reference           | Worst relative error |
| ------------------- | -------------------- |
| Numerical gradients | 8.918e-07            |
| torch.autograd      | 4.122e-14            |

40 epochs, both optimisers from the same starting weights:

| Optimiser    | Train loss      | Test accuracy |
| ------------ | --------------- | ------------- |
| SGD lr=0.5   | 1.1099 → 0.0089 | 95.8%         |
| Adam lr=0.01 | 1.1861 → 0.0032 | 97.5%         |

Stretch (Tanh + Sigmoid/MSE, odd vs even): loss 0.1482 → 0.0078, accuracy 99.4%.

[Loss curves](plots/loss_curve.png)

## Files

| File                        | What it does                                            |
| --------------------------- | ------------------------------------------------------- |
| src/layers.py               | Linear, ReLU, Sigmoid, Tanh + their backward passes     |
| src/losses.py               | SoftmaxCrossEntropy, MSELoss                            |
| src/network.py              | MLP: forward in order, backward in reverse              |
| src/optimizers.py           | SGD, Momentum, Adam                                     |
| src/data.py                 | load_digits, scale, split, minibatches                  |
| train.py                    | runs training, saves plots                              |
| gradient_check.py           | numerical + torch gradient comparison                   |
| test_correctness.py         | PASS/FAIL harness                                       |
| test_harness_sensitivity.py | proves harness catches real bugs                        |
| plots/                      | loss curves                                             |
| WRITEUP.md                  | gradient derivations, results, mistakes                 |
| predict.py                  | asks for an image and predicts using the neural network |
| requirements.txt            | Python dependencies                                     |

## Deliverables

| #   | What                            | Where                                        |
| --- | ------------------------------- | -------------------------------------------- |
| 1.1 | Feedforward net from matrix ops | src/network.py , src/layers.py               |
| 1.2 | Forward pass                    | Linear , ReLU in src/layers.py               |
| 1.3 | Manual backward + derivations   | backward() in each layer; WRITEUP §2         |
| 1.4 | Gradient check                  | test_correctness.py ; WRITEUP §3             |
| 1.5 | Train, loss decreases           | train.py ; WRITEUP §4                        |
| 1.6 | Gradient mistakes               | WRITEUP §5                                   |
|     | Tanh + Sigmoid/MSE              | src/layers.py , src/losses.py ; WRITEUP §2.5 |
|     | Momentum / Adam                 | src/optimizers.py                            |

Full write-up with derivations and mistakes: [WRITEUP.md](WRITEUP.md)
