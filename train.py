"""
Trains LOBTransformer on the exported sequences.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import LOBTransformer

# Config 
DATA_DIR        = "data"
CHECKPOINT_PATH = "best_model.pt"

BATCH_SIZE      = 128
NUM_EPOCHS      = 15
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-3

# Use nvidia gpu for training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset 

class LOBDataset(Dataset):
    def __init__(self, split: str):
        x_path   = os.path.join(DATA_DIR, f"X_{split}.npy")
        y_path   = os.path.join(DATA_DIR, f"y_{split}.npy")

        self.X = np.load(x_path)
        self.y = np.load(y_path)

        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# Class weights 

def compute_class_weights(y: np.ndarray, num_classes: int = 3, smoothing: float = 0.3) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = (total / (num_classes * counts)) ** smoothing
    return torch.tensor(weights, dtype=torch.float32)


#  Train / eval loops 

def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            if is_training:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += batch_size

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# Entry point 

def main():
    print(f"Device: {DEVICE}")

    print("\nLoading datasets...")
    train_ds = LOBDataset("train")
    val_ds   = LOBDataset("val")

    print(f"  train: {len(train_ds):,} sequences")
    print(f"  val:   {len(val_ds):,} sequences")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print("\nComputing class weights from training labels...")
    class_weights = compute_class_weights(train_ds.y).to(DEVICE)
    print(f"  weights (DOWN, FLAT, UP): {class_weights.tolist()}")

    print("\nBuilding model...")
    model = LOBTransformer().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")

    print(f"\nTraining for {NUM_EPOCHS} epochs...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer=None
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)

        flag = " <- saved (best so far)" if improved else ""
        print(
            f"Epoch {epoch:>2}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"lr={current_lr:.6f} | "
            f"{elapsed:.1f}s{flag}"
        )

    print(f"\nDone. Best model saved to {CHECKPOINT_PATH} (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()