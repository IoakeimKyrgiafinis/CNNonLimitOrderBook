import os
from dotenv import load_dotenv

load_dotenv()

# database

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5433)),
    "dbname": os.getenv("DB_NAME", "lobdb"),
    "user": os.getenv("DB_USER", "lobuser"),
    "password": os.getenv("DB_PASSWORD","lobpass"),

}

# binance

SYMBOL = "BTCUSDT"
DEPTH_LEVELS = 10
UPDATE_SPEED_MS = 100

WS_URL = (
    f"wss://stream.binance.com:9443/ws/"
    f"{SYMBOL.lower()}@depth{DEPTH_LEVELS}@{UPDATE_SPEED_MS}ms"
)

#collection

BATCH_SIZE = 50
MAX_RECONNECTS = 10
LOG_EVERY_N = 500