"""Training loops. Writes checkpoints to checkpoints/.

    python3 -m ml.train discard     # Stage 1b: hand EV + crib EV
    python3 -m ml.train pegging     # Stage 3: points taken + points given up
"""
import csv
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ml.features import encode_split, encode_pegging, PEGGING_FEATURE_SIZE
from ml.models import DiscardCribNet, PeggingNet

ROOT = "/Users/ryanwilliams/Coding/Cribbage AI"

# Pegging rows are 610 floats each, so the full 7.4M would be 18 GB before
# counting the peak during conversion. Cap it, and raise the cap only if
# validation loss is still falling when training runs out of data.
MAX_PEGGING_ROWS = 5_000_000


# ===============================================
# Loading
# ===============================================

def load_discard():
    X, y = [], []
    with open(f"{ROOT}/data/raw/discard_crib_labels.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)    # skip header
        for row in reader:
            X.append(encode_split(row[0:4], row[4:6]))
            y.append([float(row[6]), float(row[7])])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_pegging(max_rows=MAX_PEGGING_ROWS):
    """Preallocated, so peak memory is the array itself and nothing more.

    Appending to a list and calling np.array() at the end holds both at
    once - roughly double the memory, at this size the difference between
    working and thrashing.
    """
    X = np.zeros((max_rows, PEGGING_FEATURE_SIZE), dtype=np.float32)
    y = np.zeros((max_rows, 2), dtype=np.float32)

    n = 0
    with open(f"{ROOT}/data/raw/pegging_decisions.jsonl", "r") as f:
        for line in f:
            if n == max_rows:
                break
            record = json.loads(line)
            X[n] = encode_pegging(record, record["action"])
            y[n] = (record["gain"], record["give"])
            n += 1

            if n % 250_000 == 0:
                print(f"  loaded {n:,} rows", flush=True)

    # trim if the file ran out first
    return X[:n], y[:n]


def shuffle_and_split_data(X, y):
    n = len(X)
    perm = np.random.permutation(n)
    X, y = X[perm], y[perm]

    n_train = int(n * 0.8)
    n_val = int(n * 0.9)

    return (X[:n_train], y[:n_train],
            X[n_train:n_val], y[n_train:n_val],
            X[n_val:], y[n_val:])


# ===============================================
# Training
# ===============================================

def run(model, X, y, col_names, checkpoint, num_epochs, lr=1e-3, batch_size=512):
    X_train, y_train, X_val, y_val, X_test, y_test = shuffle_and_split_data(X, y)

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_val = torch.from_numpy(X_val)
    y_val = torch.from_numpy(y_val)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(X_train.shape, y_train.shape)

    best_val = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            pred = model(X_val)
            val_loss = loss_fn(pred, y_val).item()
            per_col = ((pred - y_val) ** 2).mean(dim=0)

        cols = "  ".join(f"{name} {per_col[i].item():.4f}"
                         for i, name in enumerate(col_names))
        marker = ""
        # save on improvement rather than at the end - the last epoch is
        # often not the best one
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint)
            marker = "  <- saved"

        print(f"epoch {epoch+1:>2}  train {train_loss:.4f}  val {val_loss:.4f}  {cols}{marker}")

    print(f"best val {best_val:.4f} -> {checkpoint}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "pegging"

    if which == "discard":
        X, y = load_discard()
        run(DiscardCribNet(), X, y,
            col_names=("hand", "crib"),
            checkpoint=f"{ROOT}/checkpoints/discard_crib_net.pt",
            num_epochs=12)

    elif which == "pegging":
        print("loading pegging decisions...")
        X, y = load_pegging()
        print(f"encoded {len(X):,} decisions")
        run(PeggingNet(), X, y,
            col_names=("gain", "give"),
            checkpoint=f"{ROOT}/checkpoints/pegging_net.pt",
            num_epochs=15)

    else:
        raise SystemExit("usage: python3 -m ml.train [discard|pegging]")


if __name__ == '__main__':
    main()
