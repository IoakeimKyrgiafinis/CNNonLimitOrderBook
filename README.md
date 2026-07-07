# BTC/USDT Limit Order Book Price Prediction

This project is an end to end machine learning pipeline that predicts short-term BTC/USDT price direction (High Frequency) based on live Limit Order book structure, it then tests whether this prediction is robust and whether it can be profitable considering round-trip transaction costs.

It is inspired by DeepLOB (Zhang et al., 2019), extended with a modern CNN + Transformer architecture, a SQL feature store and a cost-aware backtest

## Conclusion

Although the model finds a real, generalizable signal, it is unprofitable because of the 0.1% Binance Transaction fees applied in the High Frequency trades that occur in the virtual Backtesting.
- 68.0% Validation accuracy across all three classes (DOWN/FLAT/UP)
- 82.0% Gross win rate on the test set at 0.8 confidence threshold
- +0.0210% gross return per trade - about 9.5 times smaller than the 0.2% round-trip fee
- UP/DOWN predictions on test set are balanced (16k vs 18k)

* 65.5% of UP predictions and 62.3% of DOWN predictions were correct on fully held out test data but Binance's 0.1% fee made the predictions not profitable

## Architecture

```
Binance WebSocket (live LOB)
        │
        ▼
PostgreSQL: lob_snapshots       (raw 10-level order book, ~588k rows)
        │
        ▼   SQL window functions
PostgreSQL: lob_features        (mid_return, spread, order_imbalance,
        │                        microprice, depth_imbalance, relative_spread)
        │
        ▼   SQL rolling past/future averages
PostgreSQL: lob_labels          (−1 / 0 / +1  →  DOWN / FLAT / UP)
        │
        ▼
NumPy sequence export           (sliding windows, time-based train/val/test split)
        │
        ▼
PyTorch: CNN + Transformer      (21k parameters, GPU-accelerated)
        │
        ▼
Trading policy                  (confidence threshold → LONG / SHORT / FLAT)
        │
        ▼
Backtest                        (realistic fees, gross vs net, signal-reversal exit)
```

## Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1. Schema | db_setup.py | Creates PostgreSQL tables and indexes for raw LOB data |
| 2. Ingestion | ingest.py | Streams live LOB depth updates from Binance WebSocket, batches and writes to PostgreSQL with auto-reconnect and graceful shutdown |
| 3. Features | features_setup.py | Creates the lob_features table with a unique index |
| 4. Feature store | compute_features.py | Computes microstructure features (spread, order imbalance, microprice, depth imbalance, returns) directly in SQL using window functions |
| 5. Labels setup | labels_setup.py | Creates the lob_labels table |
| 6. Labeling | generate_labels.py | Generates DeepLOB-style 3-class labels using rolling past/future mid-price averages and a tunable threshold |
| 7. Export | export_sequences.py | Builds sliding-window sequences with a time-based (not random) train/val/test split |
| 8. Model | model.py | CNN feature extractor + Transformer encoder + classification head (~21k parameters) |
| 9. Training | train.py | Class-weighted loss, gradient clipping, LR scheduling, GPU-accelerated |
| 10. Evaluation | evaluate.py | Confusion matrix and per-class precision/recall/F1 |
| 11. Backtest | backtest.py | Confidence-thresholded policy with signal-reversal exit, realistic Binance fees, gross vs net comparison |

+ The initial model included mid_price as a feature. This caused the model to memorise price regiumes and it predicted UP 81% of the time on the test set (which was collected during a different market regime) despite 79.6% validation accuracy. By replacing mid_price with mid_return generalization was fixed and gross trading edge was tripled.

+ Time-based train/val/test split. Random splitting would leak near duplicate overlapping sequences. Splitting strictly by time helps us avoid this.

+ Gross vs net reporting. Reporting both pre-fee and post-fee performance helps us realise that while under utopic conditions the model is profitable, after applying realistic transaction fees this is not true.

## Results summary

| Metric | Value |
|---|---|
| Validation accuracy | 68.0% |
| Gross win rate (test, threshold=0.8) | 82.0% |
| Avg gross return per trade | +0.0210% |
| Round-trip fee (Binance spot taker) | 0.200% |
| Fee gap (gross edge / fee) | ~9.5x |
| UP prediction directional accuracy | 65.5% |
| DOWN prediction directional accuracy | 62.3% |
| Test set prediction collapse (mid_price model) | 81% UP |
| Test set prediction collapse (mid_return model) | None |

## Tech stack

Python · PyTorch · PostgreSQL · Docker · Binance WebSocket API · NumPy · psycopg2

```bash
# 1. Start the database
docker run --name lob-db -e POSTGRES_USER=lobuser -e POSTGRES_PASSWORD=lobpass \
  -e POSTGRES_DB=lobdb -p 5433:5432 -d postgres:16

# 2. Set up schema
python db_setup.py
python features_setup.py
python labels_setup.py

# 3. Collect data (leave running, Ctrl+C to stop)
python ingest.py

# 4. Compute features and labels
python compute_features.py
python generate_labels.py

# 5. Export sequences
python export_sequences.py

# 6. Train
python train.py

# 7. Evaluate and backtest
python evaluate.py
python backtest.py
```
