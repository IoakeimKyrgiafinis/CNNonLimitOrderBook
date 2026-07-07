"""

Reads lob_snapshots, computes microstructure features using SQL window
functions, and writes results to lob_features.

Features computed:
    mid_price       : (bid_price_1 + ask_price_1) / 2
    spread          : normalized spread (ask - bid) / mid
    order_imbalance : volume imbalance at best bid/ask level
    microprice      : deviation of volume-weighted mid from raw mid
    depth_imbalance : volume imbalance across all 10 book levels
    mid_return      : log return from previous snapshot
    relative_spread : same as spread (mapped to the relative_spread column)

Runs in batches with overlap so window functions (LAG, STDDEV)



"""

import psycopg2
from config import DB_CONFIG, SYMBOL, DEPTH_LEVELS

BATCH_SIZE = 10_000
OVERLAP    = 200


# DB 

def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn


def get_last_processed_id(cur) -> int:
    """Return the highest snapshot_id already in lob_features (0 if none)."""
    cur.execute(
        """
        SELECT COALESCE(MAX(snapshot_id), 0)
        FROM lob_features
        WHERE symbol = %s
        """,
        (SYMBOL,),
    )
    return cur.fetchone()[0]


# Helpers 

def build_depth_sum(prefix: str) -> str:
    """Sum all volume levels for a given side: bid_vol_1 + ... + bid_vol_10"""
    return " + ".join(f"{prefix}_vol_{i}" for i in range(1, DEPTH_LEVELS + 1))


# Core feature pipeline

def compute_and_insert_batch(cur, from_id: int, to_id: int):
    """
    Compute features for snapshots with id BETWEEN from_id AND to_id.

    Uses a CTE pipeline:
        base    - pulls raw LOB columns and computes depth sums
        lagged  - computes per-row features using LAG window functions
        rolling - (same as lagged here; kept for clarity if extending later)

    ON CONFLICT (snapshot_id) DO UPDATE allows safe reruns without duplicates,
    but requires a unique index on lob_features(snapshot_id) — see features_setup.py.
    """

    bid_depth = build_depth_sum("bid")
    ask_depth = build_depth_sum("ask")

    sql = f"""
    WITH base AS (
        SELECT
            id AS snapshot_id,
            exchange_ts,
            symbol,

            (bid_price_1 + ask_price_1) / 2.0  AS mid,

            bid_price_1,
            ask_price_1,
            bid_vol_1,
            ask_vol_1,

            {bid_depth} AS bid_depth,
            {ask_depth} AS ask_depth

        FROM lob_snapshots
        WHERE symbol = %s
          AND id BETWEEN %s AND %s
    ),

    lagged AS (
        SELECT
            *,

            -- log return from previous snapshot
            LN(mid / NULLIF(LAG(mid, 1) OVER w, 0))  AS ret_1,

            -- normalized spread: (ask - bid) / mid
            (ask_price_1 - bid_price_1) / NULLIF(mid, 0)  AS spread_norm,

            -- order imbalance at level 1: (bid_vol - ask_vol) / total
            (bid_vol_1 - ask_vol_1)
                / NULLIF(bid_vol_1 + ask_vol_1, 0)  AS obi_l1,

            -- depth imbalance across all levels
            (bid_depth - ask_depth)
                / NULLIF(bid_depth + ask_depth, 0)  AS obi_depth,

            -- microprice deviation from mid (volume-weighted mid - raw mid) / mid
            (
                (bid_price_1 * ask_vol_1 + ask_price_1 * bid_vol_1)
                / NULLIF(bid_vol_1 + ask_vol_1, 0)
                - mid
            ) / NULLIF(mid, 0)  AS microprice_dev

        FROM base
        WINDOW w AS (PARTITION BY symbol ORDER BY exchange_ts)
    )

    INSERT INTO lob_features (
        snapshot_id,
        exchange_ts,
        symbol,
        mid_price,
        spread,
        order_imbalance,
        microprice,
        depth_imbalance,
        mid_return,
        relative_spread
    )
    SELECT
        snapshot_id,
        exchange_ts,
        symbol,
        mid,
        spread_norm,
        obi_l1,
        microprice_dev,
        obi_depth,
        ret_1,
        spread_norm       -- relative_spread = same as normalized spread
    FROM lagged

    ON CONFLICT (snapshot_id) DO UPDATE SET
        exchange_ts     = EXCLUDED.exchange_ts,
        symbol          = EXCLUDED.symbol,
        mid_price       = EXCLUDED.mid_price,
        spread          = EXCLUDED.spread,
        order_imbalance = EXCLUDED.order_imbalance,
        microprice      = EXCLUDED.microprice,
        depth_imbalance = EXCLUDED.depth_imbalance,
        mid_return      = EXCLUDED.mid_return,
        relative_spread = EXCLUDED.relative_spread;
    """

    cur.execute(sql, (SYMBOL, from_id, to_id))


# Main loop 

def main():
    print(f"Connecting to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    conn = get_connection()

    with conn.cursor() as cur:

        last_id = get_last_processed_id(cur)   
        print(f"Last processed snapshot_id: {last_id}")

        cur.execute(
            "SELECT COALESCE(MAX(id), 0) FROM lob_snapshots WHERE symbol = %s",
            (SYMBOL,),
        )
        max_id = cur.fetchone()[0]
        print(f"Max snapshot_id in lob_snapshots: {max_id}")

        if last_id >= max_id:
            print("Nothing new to process.")
            conn.close()
            return

        from_id = last_id + 1
        total = 0

        while from_id <= max_id:
            to_id = min(from_id + BATCH_SIZE - 1 + OVERLAP, max_id)

            compute_and_insert_batch(cur, from_id, to_id)
            conn.commit()

            processed = to_id - from_id + 1
            total += processed
            print(f"Processed {from_id} → {to_id} | total: {total:,}")

            new_from_id = to_id - OVERLAP + 1
            if new_from_id <= from_id:
                break
            from_id = new_from_id

    conn.close()
    print(f"\nDone. {total:,} snapshots processed into lob_features.")


if __name__ == "__main__":
    main()