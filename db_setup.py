#database setup, run once to create SQL schema

import psycopg2
from psycopg2 import sql
from config import DB_CONFIG, DEPTH_LEVELS

#helper functions

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def level_columns(side:str, field:str, n:int) -> str:
    #generate column definitions for n price/volume levels

    return ",\n ".join(
        f"{side}_{field}_{i} NUMERIC(18,8) NOT NULL"
                       for i in range(1, n+1)
    )


#schema

def create_tables(cur):
    #snapshot table, one row per LOB update received from Binance
    cur.execute(f"""
                CREATE TABLE IF NOT EXISTS lob_snapshots (
                    id  BIGSERIAL PRIMARY KEY,
                    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    exchange_ts BIGINT NOT NULL,
                    symbol TEXT NOT NULL,
                
                
                {level_columns('bid','price', DEPTH_LEVELS)},
                {level_columns('bid','vol',DEPTH_LEVELS)},

                
                {level_columns('ask','price',DEPTH_LEVELS)},
                {level_columns('ask','vol',DEPTH_LEVELS)}
                
                );


    """ )
    

    #metadata table one row per ingestion run

    cur.execute("""
        CREATE TABLE IF NOT EXISTS lob_runs (
                run_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                ended_at TIMESTAMPTZ,
                rows_collected BIGINT NOT NULL DEFAULT 0,
                notes TEXT
                    
        );
                
    """)


def create_indexes(cur):
    #primary query pattern, fetch all snapshots for a symbol in a time window
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_lob_symbol_ts
        ON lob_snapshots (symbol,exchange_ts);
    
    """)

    #secondary, time only queries
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_lob_exchange_ts
        ON lob_snapshots (exchange_ts DESC);

    """)

    #partial index, latest snapshot per symbol
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_lob_received
        ON lob_snapshots (received_at DESC);

    """)


#entry point


def main():
    print(f"Connecting to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    conn = get_connection()
    conn.autocommit=True

    with conn.cursor() as cur:
        print('Creating tables...')
        create_tables(cur)

        print('Creating Indexes')
        create_indexes(cur)
    
    conn.close()

    print('Done. Schmema is ready')
    print(f'\nTable: lob_snapshots ({DEPTH_LEVELS*4} price/vol columns +metadata)')
    print( "Table: lob_runs (One row per ingestion session)")
    print('Index: (symbol, exchange) -primary query pattern')
    print('Index: (exchange_ts DESC) - time-window queries')
    print('Index: (received_at DESC) -latest snapshot lookups')

if __name__ == "__main__":
    main()