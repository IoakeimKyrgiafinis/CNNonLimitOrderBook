"""
Run once to create the lob_labels table.
"""

import psycopg2
from config import DB_CONFIG


def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn


def create_labels_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lob_labels (
            id              BIGSERIAL PRIMARY KEY,
            snapshot_id     BIGINT NOT NULL REFERENCES lob_snapshots(id),
            exchange_ts     BIGINT NOT NULL,
            symbol          TEXT NOT NULL,

            past_avg_mid    NUMERIC(18, 8),
            future_avg_mid  NUMERIC(18, 8),
            pct_change      NUMERIC(18, 10),

            label           SMALLINT,      -- -1, 0, +1   (NULL if window incomplete)

            k_window        INTEGER NOT NULL,     -- snapshots used per side
            threshold       NUMERIC(10, 8) NOT NULL,  -- threshold used to generate this label

            computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)


def create_indexes(cur):
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_labels_symbol_ts
        ON lob_labels (symbol, exchange_ts);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_labels_snapshot_id
        ON lob_labels (snapshot_id);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_labels_label
        ON lob_labels (label);
    """)


def main():
    print(f"Connecting to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    conn = get_connection()

    with conn.cursor() as cur:
        print("Creating lob_labels table...")
        create_labels_table(cur)

        print("Creating indexes...")
        create_indexes(cur)
        
    conn.close()
    
    print("Done. lob_labels table is ready.")


if __name__ == "__main__":
    main()