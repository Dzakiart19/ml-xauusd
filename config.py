"""
Konfigurasi global bot XAUUSD Signal
"""

import os

# ─── Deriv WebSocket ───────────────────────────────────────────────────────────
DERIV_WS_URL   = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
DERIV_SYMBOL   = "frxXAUUSD"
CANDLE_COUNT   = 500          # jumlah candle historis
GRANULARITY    = 300          # 5 menit (dalam detik)

# ─── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# File untuk menyimpan chat_id yang terdaftar
CHAT_IDS_FILE  = "chat_ids.json"

# ─── Machine Learning ─────────────────────────────────────────────────────────
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = 10
MODEL_FILE      = "model.joblib"
SCALER_FILE     = "scaler.joblib"

# Retrain setiap N trade selesai
RETRAIN_EVERY   = 50

# ─── Signal Logic ─────────────────────────────────────────────────────────────
ATR_TP_MULTIPLIER = 2.0     # TP = entry ± (ATR × 2)
ATR_SL_MULTIPLIER = 1.5     # SL = entry ± (ATR × 1.5)

# Minimum skor ensemble sebelum sinyal dikirim (dari max 10)
MIN_ENSEMBLE_SCORE = 5

# Interval cek sinyal (detik)
SIGNAL_CHECK_INTERVAL = 5

# Max candle ke depan untuk simulasi label awal
LABEL_LOOKAHEAD = 50

# ─── Database ─────────────────────────────────────────────────────────────────
DB_FILE = "trades.db"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL  = "INFO"
