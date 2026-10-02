"""One training loop for LSTM/CNN/TCN/Transformer, so they're all tuned and
early-stopped the same way."""
import copy
import time

import numpy as np
import torch

from .evaluation import compute_metrics


def iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    n = len(y)
    order = np.random.RandomState(seed).permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]


def predict_logits(model, X: np.ndarray, device, batch_size: int = 512) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(device)
            chunks.append(model(xb).cpu())
    return torch.cat(chunks)


def train_classifier(
    model, X_train, y_train, X_val, y_val, *, epochs, batch_size=256, lr=1e-3,
    device="cpu", patience=3, seed=42, verbose=True,
):
    """Adam + cross-entropy, early-stops on validation macro-F1.
    Returns (model w/ best-epoch weights, history, best_val_macro_f1, train_time_s)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1, best_state, no_improve = -1.0, None, 0
    start_time = time.time()
    y_val_t = torch.from_numpy(y_val)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, shuffle=True, seed=seed + epoch):
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)
            optimizer.zero_grad()
            logits = model(xb_t)
            loss = loss_fn(logits, yb_t)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(yb)
            n_seen += len(yb)
        train_loss = running_loss / n_seen

        val_logits = predict_logits(model, X_val, device, batch_size)
        val_loss = loss_fn(val_logits, y_val_t).item()
        val_pred = val_logits.argmax(dim=1).numpy()
        val_macro_f1 = compute_metrics(y_val, val_pred)["macro_f1"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_macro_f1)
        if verbose:
            print(f"epoch {epoch:2d}/{epochs} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_macro_f1 {val_macro_f1:.4f}")

        if val_macro_f1 > best_f1:
            best_f1, best_state, no_improve = val_macro_f1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    model.load_state_dict(best_state)
    return model, history, best_f1, time.time() - start_time
