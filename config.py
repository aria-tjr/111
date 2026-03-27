# All configuration, API keys, constants
BINANCE_API_KEY = ""
BINANCE_SECRET = ""
ALTFINS_API_KEY = ""
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

DEFAULT_SYMBOL = "ETHUSDT"
TIMEFRAMES = ["4h", "1h", "15m"]
PRIMARY_TF = "15m"

# Leverage gating
LEVERAGE_MAP = [
    (85, 100, 200),   # score 85-100 -> 200x
    (70, 84, 100),    # score 70-84 -> 100x
    (55, 69, 50),     # score 55-69 -> 50x
]
MIN_SCORE = 55

# Scanner
SCANNER_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                   "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
SCANNER_MIN_SCORE = 80
SCANNER_INTERVAL_SECONDS = 900  # 15 min

# Layer weights (sum to 100) - adaptive
LAYER_WEIGHTS_TRENDING = {
    "l1": 15,  # regime gate
    "l2": 25,  # trend - more important when trending
    "l3": 20,  # momentum
    "l4": 20,  # order flow
    "l5": 15,  # structure
    "l6": 5,   # session/correlation
}
LAYER_WEIGHTS_VOLATILE = {
    "l1": 15,
    "l2": 15,
    "l3": 15,
    "l4": 30,  # order flow more important in volatile
    "l5": 20,
    "l6": 5,
}
