""" 
Predict a handwritten digit from any image file.

    python predict.py

The script will:
  1. Ask you for the path to your image (full path, e.g. C:/Users/.../image.png).
  2. Convert that image (any size, any format, phone photo) to what the
     network needs: grayscale, 8x8, 64 numbers in [0, 1].
  3. Print the prediction.

You can keep entering paths to test multiple images. Type 'q' to quit.
Your phone photo: colour, probably large (e.g. 3000x4000), dark ink on white paper.
Network wants:    8x8 grayscale, white/bright digit on dark background, values 0-1.

Steps it performs:
  1. Open the image.
  2. Convert to grayscale (drop colour).
  3. Invert pixel values (white paper -> black, dark ink -> bright).
     Training data has 0 = background (dark), higher = digit (bright).
     Your photo is the opposite, so we invert to match.
  4. Resize to 8x8 using high-quality downsampling.
  5. Flatten to 64 numbers.
  6. Scale to [0, 1] by dividing by 255.
"""

import sys

import numpy as np
from PIL import Image

from src.data import iterate_minibatches, load_digits_split
from src.losses import SoftmaxCrossEntropy
from src.network import MLP
from src.optimizers import SGD


def train_model():
    """
    Train the digit classifier and return it ready to predict.

    Takes about 15 seconds. Trained once per run; weights stay in memory.
    """
    x_train, y_train, x_test, y_test = load_digits_split()

    rng = np.random.default_rng(0)
    model = MLP.build_classifier(
        d_in=64, d_hidden=32, d_out=10,
        loss=SoftmaxCrossEntropy(),
        rng=rng,
    )
    optimizer = SGD(model.parameters(), lr=0.5)

    print("Training the network on handwritten digits...")
    for epoch in range(40):
        for xb, yb in iterate_minibatches(x_train, y_train, 32, rng):
            model.compute_loss(xb, yb)
            model.backward()
            optimizer.step()

    test_acc = model.accuracy(x_test, y_test)
    print(f"Training done. Test accuracy on built-in digits: {test_acc:.1%}\n")
    return model

def preprocess_image(path):
    """
    Load any image and convert it to a (1, 64) array the network can digest.

    Returns the array and the 8x8 numpy grid (for the ASCII preview).

    Pipeline:
      1. Greyscale
      2. Invert  (dark ink on white paper -> bright digit on dark background)
      3. Binarize with Otsu's threshold  <- KEY FIX
         Phone photos have grey paper, shadows, noise. Without binarization the
         whole image is uniformly grey after inversion and the 8x8 grid sees
         noise everywhere instead of a clear digit shape.
      4. Auto-crop to the bounding box of the digit (removes empty border)
      5. Pad by 10% on each side so the digit does not touch the edges
         (training digits have a small margin)
      6. Resize to 8x8
      7. Normalise to [0, 1]
    """
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float64)

    #   invert so digit is bright, background is dark  
    arr = 255.0 - arr

    flat = arr.flatten()
    threshold = _otsu_threshold(flat)
    binary = np.where(arr >= threshold, 255.0, 0.0)

    #   auto-crop to digit bounding box  
    rows = np.any(binary > 0, axis=1)
    cols = np.any(binary > 0, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        binary = binary[r0:r1+1, c0:c1+1]

    #   pad 10% on each side (digit should not fill edge-to-edge)  
    h, w = binary.shape
    pad_h = max(1, int(h * 0.10))
    pad_w = max(1, int(w * 0.10))
    binary = np.pad(binary, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=0)

    #   resize to 8x8  
    padded_img = Image.fromarray(binary.astype(np.uint8))
    small = padded_img.resize((8, 8), Image.LANCZOS)

    grid = np.array(small, dtype=np.float64) / 255.0
    return grid.flatten().reshape(1, 64), grid


def _otsu_threshold(flat_pixels):
    """
    Otsu's method: find the threshold t that maximises between-class variance.
    Tries every possible value 0-255 and picks the one where the two groups
    (below t and above t) are most separated. Much better than a fixed 128
    threshold because it adapts to the actual brightness of the image.
    """
    best_t = 128
    best_var = -1.0
    total = len(flat_pixels)

    for t in range(1, 255):
        below = flat_pixels[flat_pixels < t]
        above = flat_pixels[flat_pixels >= t]
        if len(below) == 0 or len(above) == 0:
            continue
        w0 = len(below) / total
        w1 = len(above) / total
        var = w0 * w1 * (below.mean() - above.mean()) ** 2
        if var > best_var:
            best_var = var
            best_t = t

    return best_t


def ascii_preview(grid):
    """
    Print a rough ASCII picture of the 8x8 image so you can see what the
    network is actually looking at after the conversion.
    """
    chars = " .:;+=xXS#"   # 10 chars for 10 brightness levels
    print("  What the network sees (8x8):")
    print("  +" + "-" * 16 + "+")
    for row in grid:
        line = "  |"
        for val in row:
            idx = min(int(val * (len(chars) - 1)), len(chars) - 1)
            line += chars[idx] * 2
        line += "|"
        print(line)
    print("  +" + "-" * 16 + "+")

def main():
    model = train_model()

    print("=" * 50)
    print("  Handwritten digit predictor")
    print("  Type 'q' to quit.")
    print("=" * 50)

    while True:
        raw = input("\nImage path: ").strip().strip('"').strip("'")

        if raw.lower() in ("q", "quit", "exit"):
            print("Goodbye.")
            break

        if not raw:
            continue

        try:
            x, grid = preprocess_image(raw)
            ascii_preview(grid)

            prediction = model.predict(x)[0]

            # Also print the full probability breakdown so you can see how
            # confident the network is for each digit.
            scores = model.forward(x)
            exp_s = np.exp(scores - scores.max())
            probs = exp_s / exp_s.sum()
            top3 = np.argsort(probs[0])[::-1][:3]

            print(f"\n  Prediction: {prediction}")
            print("  Confidence:")
            for d in top3:
                bar = "#" * int(probs[0][d] * 20)
                print(f"    digit {d}: {probs[0][d]*100:5.1f}%  {bar}")

        except FileNotFoundError:
            print(f"  File not found: {raw}")
        except Exception as exc:
            print(f"  Error reading image: {exc}")


if __name__ == "__main__":
    main()

