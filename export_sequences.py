"""
Pulls lob_features joined with lob_labels from Postgres, builds
sliding-window sequences of length SEQ_LEN, and saves them as numpy
arrays split into train/val/test by TIME (not randomly).

Output files (in ./data/):
    X_train.npy, y_train.npy, 
    X_val.npy,   y_val.npy,   
    X_test.npy,  y_test.npy,  

X shape: (num_sequences, SEQ_LEN, num_features)
y shape: (num_sequences,)        values in {0, 1, 2} -> {DOWN, FLAT, UP}
"""

import os
import numpy as np
import psycopg2
import psycopg2.extras
from config import DB_CONFIG, SYMBOL

# Parameters 
SEQ_LEN = 100          # number of past snapshots the model sees at once

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

FEATURE_COLUMNS = [
    "mid_return",
    "spread",
    "order_imbalance",
    "microprice",
    "depth_imbalance",
    "relative_spread",
]

OUTPUT_DIR = "data"


#  Step 1: pull joined feature+label rows from Postgres 

def fetch_joined_rows(conn):
    """
    Join lob_features to lob_labels on snapshot_id, ordered by time.
    Only rows with valid labels (label IS NOT NULL) are included.
    """
    cols = ", ".join(f"f.{c}" for c in FEATURE_COLUMNS)

    sql = f"""
        SELECT
            f.snapshot_id,
            f.exchange_ts,
            {cols},
            l.label
        FROM lob_features f
        JOIN lob_labels l
          ON l.snapshot_id = f.snapshot_id
        WHERE f.symbol = %s
          AND l.label IS NOT NULL
          AND f.mid_return IS NOT NULL
        ORDER BY f.exchange_ts
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        print("Fetching joined feature+label rows from Postgres...")
        cur.execute(sql, (SYMBOL,))
        rows = cur.fetchall()

    print(f"Fetched {len(rows):,} labeled rows.")
    return rows


# Step 2: build sliding-window sequences 



def build_sequences(rows):
    num_features = len(FEATURE_COLUMNS)
    n_rows = len(rows)
    
    feature_matrix = np.array(
        [[float(r[c]) for c in FEATURE_COLUMNS] for r in rows],
        dtype=np.float32
    )
    
    label_map = {-1: 0, 0: 1, 1: 2}
    labels = np.array([label_map[r["label"]] for r in rows], dtype=np.int64)
    
    
    indices = list(range(0, n_rows - SEQ_LEN + 1))
    n_sequences = len(indices)
    
    print(f"Building {n_sequences:,} sequences (reduced from {n_rows - SEQ_LEN + 1:,} ")
    
    X = np.zeros((n_sequences, SEQ_LEN, num_features), dtype=np.float32)
    y = np.zeros(n_sequences, dtype=np.int64)
    
    for i, start_idx in enumerate(indices):
        X[i] = feature_matrix[start_idx : start_idx + SEQ_LEN]
        y[i] = labels[start_idx + SEQ_LEN - 1]
    
    return X, y

# Step 3: time-based split 

def time_split(X, y):
    n = len(X)
    train_end = int(n * TRAIN_FRAC)
    val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

    splits = {
        "train": (X[:train_end], y[:train_end]),
        "val":   (X[train_end:val_end], y[train_end:val_end]),
        "test":  (X[val_end:], y[val_end:]),
    }

    print("\n--- Split sizes ---")
    for name, (Xs, ys) in splits.items():
        print(f"  {name:<6} X={Xs.shape}  y={ys.shape}")

    return splits


def print_class_balance(splits):
    print("\n--- Class balance per split ---")
    names = {0: "DOWN", 1: "FLAT", 2: "UP"}
    for split_name, (Xs, ys) in splits.items():
        counts = np.bincount(ys, minlength=3)
        pct = counts / counts.sum() * 100
        line = "  ".join(f"{names[i]}={counts[i]:,} ({pct[i]:.1f}%)" for i in range(3))
        print(f"  {split_name:<6} {line}")


#  Step 4: save to disk 

def save_splits(splits):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, (Xs, ys) in splits.items():
        np.save(os.path.join(OUTPUT_DIR, f"X_{name}.npy"), Xs)
        np.save(os.path.join(OUTPUT_DIR, f"y_{name}.npy"), ys)


# Entry point 

def main():
    print(f"Connecting to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    conn = psycopg2.connect(**DB_CONFIG)

    rows = fetch_joined_rows(conn)
    conn.close()

    X, y = build_sequences(rows)
    print(f"\nFinal X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")

    splits = time_split(X, y)
    print_class_balance(splits)
    save_splits(splits)

    print("\nDone. Sequences ready for PyTorch training.")


if __name__ == "__main__":
    main()