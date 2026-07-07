import numpy as np
import torch
from torch.utils.data import DataLoader

from model import LOBTransformer
from train import LOBDataset, DATA_DIR, BATCH_SIZE, CHECKPOINT_PATH, DEVICE


CONFIDENCE_THRESHOLD = 0.8    # direction confidence
MAX_HOLD_PERIOD      = 400    # seconds at 10 Hz
FEE_RATE             = 0.001  # 0.1% per side (Binance taker)

CLASS_NAMES = ["DOWN", "FLAT", "UP"]


# 1. Inference

def get_outputs(model, loader):
    """Return (probs) for entire dataset."""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def get_mid_prices(X):
    return X[:, -1, 0]   # last timestep, feature 0 = mid_price


# 2. Signal Generation

def generate_signals(probs):
    """
    Direction‑only signals for K = 400.

    Trade only when:
      - Direction confidence >= CONFIDENCE_THRESHOLD
      - Class is UP or DOWN (ignore FLAT)
    """
    max_probs = probs.max(axis=1)
    predicted_class = probs.argmax(axis=1)

    signals = np.zeros(len(probs), dtype=np.int64)

    confident = max_probs >= CONFIDENCE_THRESHOLD

    long_cond  = confident & (predicted_class == 2)  # UP
    short_cond = confident & (predicted_class == 0)  # DOWN

    signals[long_cond]  =  1
    signals[short_cond] = -1

    return signals


# 3. Backtest Engine (lookahead‑safe)

def run_backtest(signals, mid_prices):
    n = len(signals)
    trade_returns = []
    trade_gross_returns = []
    trade_directions = []
    trade_hold_lengths = []

    i = 0
    while i < n - MAX_HOLD_PERIOD - 2:

        if signals[i] != 0:
            direction = signals[i]
            opposite  = -direction

            entry_price = mid_prices[i + 1]  # avoid lookahead bias

            exit_idx = None
            for j in range(i + 1, min(i + 1 + MAX_HOLD_PERIOD, n - 2)):
                if signals[j] == opposite:
                    exit_idx = j
                    break

            if exit_idx is None:
                exit_idx = min(i + MAX_HOLD_PERIOD, n - 2)

            exit_price = mid_prices[exit_idx + 1]
            hold_length = exit_idx - i

            if direction == 1:
                raw_return = (exit_price - entry_price) / entry_price
            else:
                raw_return = (entry_price - exit_price) / entry_price

            fee = FEE_RATE * 2
            net_return = raw_return - fee

            trade_returns.append(net_return)
            trade_gross_returns.append(raw_return)
            trade_directions.append(direction)
            trade_hold_lengths.append(hold_length)

            i = exit_idx
        else:
            i += 1

    return (
        np.array(trade_returns),
        np.array(trade_gross_returns),
        np.array(trade_directions),
        np.array(trade_hold_lengths),
    )


# 4. Metrics

def compute_metrics(trade_returns, trade_gross_returns, trade_directions, trade_hold_lengths):
    n_trades = len(trade_returns)

    if n_trades == 0:
        print("No trades triggered at this confidence threshold.")
        return

    total_return = trade_returns.sum()
    avg_return = trade_returns.mean()
    win_rate = (trade_returns > 0).mean()

    gross_total = trade_gross_returns.sum()
    gross_avg = trade_gross_returns.mean()
    gross_win_rate = (trade_gross_returns > 0).mean()

    std_return = trade_returns.std()
    sharpe = avg_return / std_return if std_return > 0 else 0.0

    cumulative = np.cumsum(trade_returns)
    running_max = np.maximum.accumulate(cumulative)
    max_drawdown = (cumulative - running_max).min()

    n_long = (trade_directions == 1).sum()
    n_short = (trade_directions == -1).sum()

    print("\n--- Backtest results ---")
    print(f"Confidence threshold : {CONFIDENCE_THRESHOLD}")
    
    print(f"Trades taken         : {n_trades:,} (LONG={n_long:,}, SHORT={n_short:,})")
    print(f"Avg hold length      : {trade_hold_lengths.mean():.1f} snapshots")
    print(f"Forced exits         : {(trade_hold_lengths == MAX_HOLD_PERIOD).sum():,}")
    print()
    print(f"{'':20}{'GROSS':>15}{'NET':>15}")
    print(f"{'Win rate':<20}{gross_win_rate:>14.1%}{win_rate:>15.1%}")
    print(f"{'Total return':<20}{gross_total:>14.4%}{total_return:>15.4%}")
    print(f"{'Avg/trade':<20}{gross_avg:>14.4%}{avg_return:>15.4%}")
    print()
    print(f"Sharpe ratio         : {sharpe:.3f}")
    print(f"Max drawdown         : {max_drawdown:.4%}")

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "total_return": total_return,
        "avg_return": avg_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "cumulative": cumulative,
    }


# 5. Entry point

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    model = LOBTransformer().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    print("Loading TEST set...")
    test_ds = LOBDataset("test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Running inference on {len(test_ds):,} sequences...")
    probs = get_outputs(model, test_loader)
    all_mid_prices = get_mid_prices_from_db()
    n_total = len(all_mid_prices)
    test_start = int(n_total * 0.85)
    mid_prices = all_mid_prices[test_start:]

    print("Generating signals...")
    signals = generate_signals(probs)

    print(f"Signals: LONG={(signals==1).sum():,} | SHORT={(signals==-1).sum():,} | FLAT={(signals==0).sum():,}")

    print("\nRunning backtest...")
    trade_returns, trade_gross_returns, trade_directions, trade_hold_lengths = run_backtest(signals, mid_prices)

    compute_metrics(trade_returns, trade_gross_returns, trade_directions, trade_hold_lengths)


def get_mid_prices_from_db():
    import psycopg2
    from config import DB_CONFIG, SYMBOL
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
        prices = np.array([row[0] for row in cur.fetchall()], dtype=np.float32)
    conn.close()
    return prices

if __name__ == "__main__":
    main()