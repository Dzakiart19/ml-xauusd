"""
Konfigurasi global bot XAUUSD Signal
"""

import os

# ─── Deriv WebSocket ───────────────────────────────────────────────────────────
DERIV_WS_URL   = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
DERIV_SYMBOL   = "frxXAUUSD"
CANDLE_COUNT   = 5000         # Deriv API maks 5000 candle @5-menit (~25 hari kalender / ~18 hari trading)
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
FEATURE_IMPORTANCE_FILE = "feature_importance.json"  # log feature importance tiap retrain

# Retrain setiap N trade selesai (fallback; jadwal adaptif diatur di signal_generator)
RETRAIN_EVERY   = 50

# ─── Signal Logic ─────────────────────────────────────────────────────────────
ATR_TP_MULTIPLIER = 2.5     # TP = entry ± (ATR × 2.5)  → R:R = 2.5:1
ATR_SL_MULTIPLIER = 1.0     # SL = entry ± (ATR × 1.0)

# ATR minimum — jangan trade saat pasar terlalu flat/choppy
ATR_MIN_THRESHOLD = 0.5

# Estimasi spread XAUUSD (dollar) — diperhitungkan di label simulasi backtest & ML
# Spread tipikal $0.30–$1.00; $0.50 adalah estimasi konservatif yang aman
SPREAD_ESTIMATE = 0.50

# Minimum rasio skor ensemble (dari total vote yang aktif)
# 0.70 = butuh 70% suara majority sebelum sinyal dikirim
MIN_ENSEMBLE_RATIO = 0.70

# ─── Filter konfirmasi arah sinyal ────────────────────────────────────────────
BUY_RSI_MAX    = 45   # BUY hanya saat RSI benar-benar oversold (< 45)
SELL_RSI_MIN   = 25   # SELL hanya jika RSI tidak extreme oversold
SELL_STOCH_MIN = 20   # SELL hanya jika stoch tidak di zona oversold

# Minimum probabilitas ML untuk konfirmasi sinyal
ML_PROBA_THRESHOLD = 0.58

# Interval cek sinyal (detik)
SIGNAL_CHECK_INTERVAL = 10

# Max candle ke depan untuk simulasi label
LABEL_LOOKAHEAD = 50

# Candle terakhir yang dikecualikan dari backtest training (Fix 1 — cegah data leakage)
# ML tidak dilatih dari candle yang akan segera dipakai untuk sinyal live
BACKTEST_HOLDOUT = 500

# Maksimum durasi trade aktif sebelum force-close timeout (Fix 3)
# 60 candle × 5 menit = 5 jam
MAX_TRADE_CANDLES = 60

# Jumlah candle terakhir yang dipakai untuk kalkulasi indikator inkremental (Fix 7)
# 600 candle cukup untuk semua indikator period ≤ 200 + buffer konvergensi EWM
INDICATOR_WINDOW = 600

# ─── Database ─────────────────────────────────────────────────────────────────
DB_FILE = "trades.db"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL  = "INFO"
