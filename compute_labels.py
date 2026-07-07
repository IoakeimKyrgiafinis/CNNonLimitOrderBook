import psycopg2
from config import DB_CONFIG, SYMBOL

# Tunable parameters 
K_WINDOW  = 50        # second horizon at 10 Hz
THRESHOLD = 0.00005  # direction threshold


# 1. DB connection

def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn




# 2. Clear old labels

def clear_existing_labels(cur):
    cur.execute("DELETE FROM lob_labels WHERE symbol = %s", (SYMBOL,))


# 3. Compute labels

def compute_labels(cur):
    sql = """
        INSERT INTO lob_labels (
            snapshot_id, exchange_ts, symbol,
            past_avg_mid, future_avg_mid, pct_change,
            label, k_window, threshold
        )
        SELECT
            snapshot_id,
            exchange_ts,
            symbol,
            past_avg_mid,
            future_avg_mid,
            pct_change,

            CASE
                WHEN pct_change >  %(threshold)s THEN  1
                WHEN pct_change < -%(threshold)s THEN -1
                WHEN pct_change IS NOT NULL      THEN  0
                ELSE NULL
            END AS label,

            %(k_window)s,
            %(threshold)s

        FROM (
            SELECT
                snapshot_id,
                exchange_ts,
                symbol,
                mid_price,

                AVG(mid_price) OVER (
                    PARTITION BY symbol
                    ORDER BY exchange_ts
                    ROWS BETWEEN %(k_window)s PRECEDING AND 1 PRECEDING
                ) AS past_avg_mid,

                AVG(mid_price) OVER (
                    PARTITION BY symbol
                    ORDER BY exchange_ts
                    ROWS BETWEEN 1 FOLLOWING AND %(k_window)s FOLLOWING
                ) AS future_avg_mid

            FROM lob_features
            WHERE symbol = %(symbol)s
        ) windowed
        CROSS JOIN LATERAL (
            SELECT
                CASE
                    WHEN past_avg_mid IS NULL OR past_avg_mid = 0 THEN NULL
                    ELSE (future_avg_mid - past_avg_mid) / past_avg_mid
                END AS pct_change
        ) calc;
    """

    cur.execute(sql, {
        "threshold": THRESHOLD,
        "k_window": K_WINDOW,
        "symbol": SYMBOL,
    })


# 4. Print distribution

def print_distribution(cur):
    cur.execute("""
        SELECT
            label,
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
        FROM lob_labels
        WHERE symbol = %s
        GROUP BY label
        ORDER BY label;
    """, (SYMBOL,))

    rows = cur.fetchall()
    print("\n--- Label distribution ---")
    for label, count, pct in rows:
        name = {1: "UP (+1)", 0: "FLAT (0)", -1: "DOWN (-1)", None: "NULL"}.get(label, str(label))
        bar = "█" * int((pct or 0) / 2)
        print(f"  {name:<18} {count:>8,}  {pct:>5}%  {bar}")


# 5. Main

def main():
    print(f"Connecting to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    print(f"K_WINDOW={K_WINDOW} | THRESHOLD={THRESHOLD}")

    conn = get_connection()

    with conn.cursor() as cur:
        

        print("Clearing old labels...")
        clear_existing_labels(cur)
        conn.commit()

        print("Computing labels...")
        compute_labels(cur)
        conn.commit()

        print_distribution(cur)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()