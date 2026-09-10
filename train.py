"""
training 
    python train.py           # both
    python train.py --main    # digits only
    python train.py --stretch # odd/even only

digits: Linear(64,32) -> ReLU -> Linear(32,10) -> Softmax+CE  (SGD and Adam)
stretch: Linear(64,16) -> Tanh -> Linear(16,1) -> Sigmoid+MSE
"""

import argparse

import numpy as np

from src.data import iterate_minibatches, load_digits_split, make_binary_parity_split
from src.layers import Linear, Sigmoid, Tanh
from src.losses import MSELoss, SoftmaxCrossEntropy
from src.network import MLP
from src.optimizers import SGD, Adam

PLOTS_DIR = "plots"


def train_classifier(optimizer_name, epochs=40, batch_size=32, lr=None, seed=0):
    # same seed for both optimisers so the curves are comparable
    x_train, y_train, x_test, y_test = load_digits_split(seed=seed)

    rng = np.random.default_rng(seed)
    model = MLP.build_classifier(
        d_in=x_train.shape[1], d_hidden=32, d_out=10, loss=SoftmaxCrossEntropy(), rng=rng
    )

    if optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), lr=0.5 if lr is None else lr)
    else:
        optimizer = Adam(model.parameters(), lr=1e-2 if lr is None else lr)

    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}

    for epoch in range(1, epochs + 1):
        batch_losses = []
        for x_batch, y_batch in iterate_minibatches(x_train, y_train, batch_size, rng):
            loss, _ = model.compute_loss(x_batch, y_batch)
            model.backward()
            optimizer.step()
            batch_losses.append(loss)

        train_loss = float(np.mean(batch_losses))
        test_loss, _ = model.compute_loss(x_test, y_test)
        test_accuracy = model.accuracy(x_test, y_test)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:>3}  train loss {train_loss:.4f}"
                f"   test loss {test_loss:.4f}   test accuracy {test_accuracy:.4f}"
            )

    return model, history, (x_train, y_train, x_test, y_test)


def train_stretch(epochs=60, batch_size=32, lr=0.5, seed=0):
    # tanh + sigmoid/mse. no cancel like softmax+ce, so needs more epochs
    x_train, y_train, x_test, y_test = make_binary_parity_split(seed=seed)

    rng = np.random.default_rng(seed)
    model = MLP(
        layers=[
            Linear(x_train.shape[1], 16, rng=rng),
            Tanh(),
            Linear(16, 1, rng=rng),
            Sigmoid(),
        ],
        loss=MSELoss(),
    )
    optimizer = SGD(model.parameters(), lr=lr)

    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}

    for epoch in range(1, epochs + 1):
        batch_losses = []
        for x_batch, y_batch in iterate_minibatches(x_train, y_train, batch_size, rng):
            loss, _ = model.compute_loss(x_batch, y_batch)
            model.backward()
            optimizer.step()
            batch_losses.append(loss)

        train_loss = float(np.mean(batch_losses))
        test_loss, predictions = model.compute_loss(x_test, y_test)
        # sigmoid is a probability, threshold at 0.5
        test_accuracy = float(((predictions >= 0.5).astype(float) == y_test).mean())

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:>3}  train loss {train_loss:.4f}"
                f"   test loss {test_loss:.4f}   test accuracy {test_accuracy:.4f}"
            )

    return model, history


def save_plot(histories, filename, title):
    # skip if matplotlib missing
    import os

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  matplotlib not installed, skipping {filename}")
        return None

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)

    figure, (ax_loss, ax_accuracy) = plt.subplots(1, 2, figsize=(11, 4))

    for label, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        ax_loss.plot(epochs, history["train_loss"], label=f"{label} train")
        ax_loss.plot(epochs, history["test_loss"], linestyle="--", label=f"{label} test")
        ax_accuracy.plot(epochs, history["test_accuracy"], label=label)

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_accuracy.set_xlabel("epoch")
    ax_accuracy.set_ylabel("test accuracy")
    ax_accuracy.set_title("Test accuracy")
    ax_accuracy.legend()
    ax_accuracy.grid(alpha=0.3)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)

    print(f"  saved {path}")
    return path


def run_main_experiment():
    print("=" * 68)
    print("Deliverable 1.5 - digits classification, 64 -> 32 -> 10")
    print("=" * 68)

    histories = {}
    for optimizer_name in ("sgd", "adam"):
        print(f"\n{optimizer_name.upper()}:")
        _, history, _ = train_classifier(optimizer_name)
        histories[optimizer_name.upper()] = history

    print("\nSummary")
    for name, history in histories.items():
        first, last = history["train_loss"][0], history["train_loss"][-1]
        print(
            f"  {name:<5} train loss {first:.4f} -> {last:.4f}"
            f"   final test accuracy {history['test_accuracy'][-1]:.4f}"
        )

    save_plot(histories, "loss_curve.png", "Digits classification (deliverable 1.5)")
    return histories


def run_stretch_experiment():
    print("\n" + "=" * 68)
    print("Stretch goal - Tanh hidden + Sigmoid/MSE output, odd-vs-even")
    print("=" * 68 + "\n")

    _, history = train_stretch()

    first, last = history["train_loss"][0], history["train_loss"][-1]
    print(
        f"\n  MSE loss {first:.4f} -> {last:.4f}"
        f"   final test accuracy {history['test_accuracy'][-1]:.4f}"
    )

    save_plot(
        {"Tanh + Sigmoid/MSE": history},
        "stretch_loss_curve.png",
        "Stretch goal: Tanh + Sigmoid/MSE, odd vs even",
    )
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", action="store_true", help="digits classifier only")
    parser.add_argument("--stretch", action="store_true", help="stretch experiment only")
    args = parser.parse_args()

    run_everything = not (args.main or args.stretch)

    if args.main or run_everything:
        run_main_experiment()
    if args.stretch or run_everything:
        run_stretch_experiment()


if __name__ == "__main__":
    main()
