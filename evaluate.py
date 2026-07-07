"""

Loads the best saved model checkpoint and evaluates it properly on
the validation set: confusion matrix + per-class precision/recall/f1.

"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import LOBTransformer
from train import LOBDataset, DATA_DIR, BATCH_SIZE, CHECKPOINT_PATH, DEVICE


CLASS_NAMES = ["DOWN", "FLAT", "UP"]


def get_predictions(model, loader):
    """Run the model over a full loader, return (all_preds, all_labels) as numpy arrays."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(y_batch.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """
    Build a num_classes x num_classes confusion matrix.
    Rows = true label, columns = predicted label.
    cm[i, j] = number of samples with true label i predicted as j.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(labels, preds):
        cm[true, pred] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray):
    print("\n--- Confusion matrix (rows=true, cols=predicted) ---")
    header = "        " + "".join(f"{name:>10}" for name in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "".join(f"{cm[i, j]:>10,}" for j in range(len(CLASS_NAMES)))
        print(f"{name:>8}{row}")


def print_per_class_metrics(cm: np.ndarray):
    print("\n--- Per-class metrics ---")
    print(f"{'class':<8}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>12}")

    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"{name:<8}{precision:>12.3f}{recall:>12.3f}{f1:>12.3f}{support:>12,}")


def print_prediction_distribution(preds: np.ndarray):
    print("\n--- What the model actually predicted (all val samples) ---")
    counts = np.bincount(preds, minlength=3)
    pct = counts / counts.sum() * 100
    for i, name in enumerate(CLASS_NAMES):
        bar = "#" * int(pct[i] / 2)
        print(f"  {name:<6} predicted {counts[i]:>7,} times ({pct[i]:>5.1f}%)  {bar}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    model = LOBTransformer().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    print("Loading validation set...")
    val_ds = LOBDataset("val")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Running inference on {len(val_ds):,} validation sequences...")
    preds, labels = get_predictions(model, val_loader)

    overall_acc = (preds == labels).mean()
    print(f"\nOverall accuracy: {overall_acc:.3f}")

    cm = confusion_matrix(preds, labels)
    print_confusion_matrix(cm)
    print_per_class_metrics(cm)
    print_prediction_distribution(preds)


if __name__ == "__main__":
    main()