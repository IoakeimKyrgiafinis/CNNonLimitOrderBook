"""

Run once to create the lob_features table.

"""

import psycopg2
from config import DB_CONFIG, DEPTH_LEVELS


def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn


def create_features_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lob_features (
            id              BIGSERIAL PRIMARY KEY,
            snapshot_id     BIGINT NOT NULL REFERENCES lob_snapshots(id),
            exchange_ts     BIGINT NOT NULL,
            symbol          TEXT NOT NULL,

            -- core microstructure features
            mid_price       NUMERIC(18, 8) NOT NULL,
            spread          NUMERIC(18, 8) NOT NULL,
            order_imbalance NUMERIC(10, 8) NOT NULL,   -- range [-1, 1]
            microprice      NUMERIC(18, 8) NOT NULL,
            depth_imbalance NUMERIC(10, 8) NOT NULL,   -- range [-1, 1]

            -- return features
            mid_return      NUMERIC(18, 10),            -- NULL for first row

            -- spread relative to mid (useful for normalization)
            relative_spread NUMERIC(18, 10) NOT NULL,

            -- computed at
            computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)


def create_indexes(cur):
    # primary query pattern for model training
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_feat_symbol_ts
        ON lob_features (symbol, exchange_ts);
    """)

    # join back to snapshots
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_feat_snapshot_id
        ON lob_features (snapshot_id);
    """)


def main():
    print(f"Connecting to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    conn = get_connection()

    with conn.cursor() as cur:
        print("Creating lob_features table...")
        create_features_table(cur)

        print("Creating indexes...")
        create_indexes(cur)

    conn.close()
    print("Done. lob_features table is ready.")


if __name__ == "__main__":
    main()
