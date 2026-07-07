

import numpy as np
import torch
from torch.utils.data import DataLoader
from config import DB_CONFIG, SYMBOL
from model import LOBTransformer
from train import LOBDataset, BATCH_SIZE, CHECKPOINT_PATH, DEVICE
import psycopg2

HOLD_PERIOD = 50

model = LOBTransformer().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

test_ds = LOBDataset("test")
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

all_probs = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
probs = np.concatenate(all_probs, axis=0)

conn = psycopg2.connect(**DB_CONFIG)
with conn.cursor() as cur:
    cur.execute("""
        SELECT f.mid_price
        FROM lob_features f
        JOIN lob_labels l ON l.snapshot_id = f.snapshot_id
        WHERE f.symbol = %s
          AND l.label IS NOT NULL
          AND f.mid_return IS NOT NULL
        ORDER BY f.exchange_ts
    """, (SYMBOL,))
    all_prices = np.array([row[0] for row in cur.fetchall()], dtype=np.float32)
conn.close()

n_total = len(all_prices)
test_start = int(n_total * 0.85)
mid_prices = all_prices[test_start:]
predicted_class = probs.argmax(axis=1)

# For every sequence where model predicts UP (class 2), check actual forward return
n = len(predicted_class)
up_idx = np.where(predicted_class == 2)[0]
up_idx = up_idx[up_idx < n - HOLD_PERIOD]

down_idx = np.where(predicted_class == 0)[0]
down_idx = down_idx[down_idx < n - HOLD_PERIOD]

up_forward_returns = (mid_prices[up_idx + HOLD_PERIOD] - mid_prices[up_idx]) / mid_prices[up_idx]
down_forward_returns = (mid_prices[down_idx + HOLD_PERIOD] - mid_prices[down_idx]) / mid_prices[down_idx]

print(f"When model predicts UP   (n={len(up_idx):,}): avg forward return = {up_forward_returns.mean():.6%}")
print(f"When model predicts DOWN (n={len(down_idx):,}): avg forward return = {down_forward_returns.mean():.6%}")
print()
print(f"UP predictions, % actually positive return:   {(up_forward_returns > 0).mean():.2%}")
print(f"DOWN predictions, % actually negative return: {(down_forward_returns < 0).mean():.2%}")

# also check raw labels vs raw mid price moves, independent of model
y_test = test_ds.y
label_up_idx = np.where(y_test == 2)[0]
label_up_idx = label_up_idx[label_up_idx < n - HOLD_PERIOD]
label_up_forward = (mid_prices[label_up_idx + HOLD_PERIOD] - mid_prices[label_up_idx]) / mid_prices[label_up_idx]
print()
print(f"Sanity check on LABELS (not model) - when true label is UP: avg forward return = {label_up_forward.mean():.6%}")
print(f"  (this should be POSITIVE if labels are correct)")