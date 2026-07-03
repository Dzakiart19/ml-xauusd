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
RF_N_ESTIMATORS = 150         # lebih banyak tree → lebih stabil
RF_MAX_DEPTH    = 8           # lebih dangkal → kurangi overfitting
MODEL_FILE      = "model.joblib"
SCALER_FILE     = "scaler.joblib"

# Retrain setiap N trade selesai
RETRAIN_EVERY   = 50

# ─── Signal Logic ─────────────────────────────────────────────────────────────
ATR_TP_MULTIPLIER = 2.5     # TP = entry ± (ATR × 2.5)  ← R:R lebih baik
ATR_SL_MULTIPLIER = 1.0     # SL = entry ± (ATR × 1.0)  ← R:R = 2.5:1

# ATR minimum — jangan trade saat pasar terlalu flat/choppy
ATR_MIN_THRESHOLD = 0.5

# Minimum rasio skor ensemble (dari total vote yang aktif)
# 0.65 = butuh 65% suara majority sebelum sinyal dikirim
MIN_ENSEMBLE_RATIO = 0.65

# Minimum probabilitas ML untuk konfirmasi sinyal
ML_PROBA_THRESHOLD = 0.58

# Interval cek sinyal (detik) — dinaikkan agar tidak overlap dengan kalkulasi indikator
SIGNAL_CHECK_INTERVAL = 10

# Max candle ke depan untuk simulasi label awal
LABEL_LOOKAHEAD = 50

# ─── Database ─────────────────────────────────────────────────────────────────
DB_FILE = "trades.db"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL  = "INFO"
