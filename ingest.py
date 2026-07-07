# this file connects to BINANCE websocket, pulls LOB snapshots and writes them to postgreSQL in batches


import json
import time
import uuid
import signal
import sys
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
import websocket

from config import (DB_CONFIG, SYMBOL, DEPTH_LEVELS, BATCH_SIZE, MAX_RECONNECTS,
                    LOG_EVERY_N,WS_URL)

#globals

batch = []   # in-memory buffer before db insert
rows_total = 0 #rows inserted this run
reconnects = 0 #reconnection counter
run_id = str(uuid.uuid4())
conn= None #psycopg2 connection
running=True #gets set to false when Ctrl+C

# helpers

def get_connection():
    c = psycopg2.connect(**DB_CONFIG)
    c.autocommit = False
    return c

def ensure_connection():
    #return existing connection or reconnect if dropped

    global conn
    try:
        if conn is None or conn.closed:
            conn = get_connection()
        #lightweight ping
        conn.cursor().execute('SELECT 1')
    except Exception:
        conn = get_connection()

    return conn

def insert_batch(records:list):
    #bulk insert list of snapshot dicts in lob_snapshots
    if not records:
        return
    
    #build column list dynamically from DEPTH_LEVELS

    side_cols = []
    for side in ('bid','ask'):
        for field in ('price','vol'):
            for i in range (1,DEPTH_LEVELS+1):
                side_cols.append(f'{side}_{field}_{i}')

    columns = ['exchange_ts','symbol'] + side_cols

    values = []

    for r in records:
        row = [r['exchange_ts'], r['symbol']]
        for col in side_cols:
            row.append(r[col])

        values.append(row)

    sql = f""" 
        INSERT INTO lob_snapshots ({', '.join(columns)})
        VALUES %s
    """

    c = ensure_connection()
    with c.cursor() as cur:
        psycopg2.extras.execute_values(cur,sql,values, page_size =BATCH_SIZE)
        
    c.commit()


def start_run():
    #insert row in lob_runs when ingestion starts
    
    c = ensure_connection()
    
    with c.cursor() as cur:
        cur.execute("""
            INSERT INTO lob_runs (run_id, symbol, started_at, rows_collected)
            VALUES (%s, %s, NOW(), 0)
            """, (run_id, SYMBOL)    
            )
        
    c.commit()
    print(f'[{ts()}] Run started | id = {run_id}')


def finish_run():
    #update lob_runs with final row count and end time
    try:
        c= ensure_connection()
        with c.cursor as cur:
            cur.execute("""
                UPDATE lob_runs
                SET ended at = NOW(), rows_collected = %s
                WHERE run_id = %s
                    
            """, (rows_total, run_id))
        c.commit()
        print(f'[{ts()}] Run finished | rows={rows_total}')
    except Exception as e:
        print(f'[{ts()}] Could not update lob_runs')

#parsing

def parse_message(raw:str) -> dict | None :
    """
    Parse a binance depth update message into a flat dict
    Binance sends
    {
                "E": < event time ms>
                "b":[['price','qty'],...], #bids
                "a":[['price','qty'],...]   #asks    
    
    }
    
    """
    try:
        msg = json.loads(raw)
        
        bids = msg.get('bids',[])
        asks = msg.get('asks',[])

        #skip if incomplete (binance sometimes sends partial updates)
        if len(bids) < DEPTH_LEVELS or len(asks) < DEPTH_LEVELS:
            return None
        
        record = {
            "exchange_ts": msg['lastUpdateId'],
            "symbol": SYMBOL,
        }

        for i, (price, qty) in enumerate (bids[:DEPTH_LEVELS], start=1):
            record[f'bid_price_{i}'] = float(price)
            record[f'bid_vol_{i}'] = float(qty)

        for i, (price, qty) in enumerate (asks[:DEPTH_LEVELS],start=1):
            record[f'ask_price_{i}'] = float(price)
            record[f'ask_vol_{i}'] = float(qty)

        return record
    
    except Exception as e:
        print(f'[{ts()}] Parse error: {e}')
        return None
    

#websocket callbacks

def on_message(ws, message):
    global batch,rows_total
    
    record=parse_message(message)
    if record is None:
        return
    
    batch.append(record)

    if len(batch) >= BATCH_SIZE:
        try:
            insert_batch(batch)
            rows_total += len(batch)
            batch = []

            if rows_total % LOG_EVERY_N < BATCH_SIZE:
                print(f'[{ts()}] Rows inserted: {rows_total:,}')

        except Exception as e:
            print(f'[{ts()}], Insert error {e}')
            #keep batch in memory, retry in next cycle
        
def on_error(ws, error):
    print(f'[{ts()}] Websocket error: {error}')


def on_close(ws, close_status_code,close_msg):
    print(f'[{ts()}] Websocket closed | status = {close_status_code}')

def on_open(ws):
    print(f'[{ts()}] Websocket Connected -> {WS_URL}')

#reconnect loop

def run_with_reconnect():
    global reconnects,running
    
    while running and reconnects <= MAX_RECONNECTS:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close

            )
            ws.run_forever(ping_interval=20,ping_timeout=10)
        except Exception as e:
            print(f'[{ts()}] Connection failed {e}')
        
        if not running:
            break

        reconnects +=1

        if reconnects >= MAX_RECONNECTS:
            print(f"[{ts()}] Max reconnects reached. Stopping")
            break


        wait = min(2 ** reconnects, 60)
        print(f"[{ts()}] Reconnecting in {wait}s... (attempt {reconnects}/{MAX_RECONNECTS})")
        time.sleep(wait)

#shutdown

def handle_shutdown(sig, frame):
    global running
    print(f'\n[{ts()}] Shutting down gracefully...')
    running = False

    #flush remaining batch
    if batch:
        print(f'[{ts()}] Flushing {len(batch)} remaining rows...')
        try:
            insert_batch(batch)
        except Exception as e:
            print(f"[{ts()}] Flush error: {e}")

    
    finish_run()
    if conn and not conn.closed:
        conn.close()
    sys.exit(0)

#utilities

def ts() -> str:
    'Current UTC time string for logging.'
    return datetime.now(timezone.utc).strftime('%H:%M:%S')


#entry point

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    print(f'[{ts()}] Starting LOB ingestion | symbol = {SYMBOL} | levels = {DEPTH_LEVELS}')
    print(f'[{ts()}] Batch size = {BATCH_SIZE} | URL = {WS_URL}')

    ensure_connection()
    start_run()
    run_with_reconnect()    
  