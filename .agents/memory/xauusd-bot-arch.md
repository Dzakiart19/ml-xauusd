---
name: XAUUSD Bot Architecture
description: Threading model, shared state pattern, dan keputusan desain utama bot sinyal XAUUSD Telegram
---

## Threading Model
5 thread:
- `Main` — Telegram polling (blocking)
- `DerivWS` — WebSocket ke Deriv API, reconnect dengan exponential backoff
- `SignalGen` — loop setiap SIGNAL_CHECK_INTERVAL detik
- `OnReadyCB` — dijalankan sekali saat data historis pertama tiba
- `Retrain` — auto-retrain dengan jadwal adaptif (setiap 10/25/50 live trade)

## Shared State
- `shared_state` dict dilindungi `state_lock` (threading.Lock)
- Key: `candles`, `candles_with_indicators`, `candles_dirty`, `current_price`, `current_bid`, `current_ask`, `is_ready`
- `candles_dirty` hanya di-set True saat candle BARU (bukan update candle forming)

## Keputusan Desain Penting
- Sinyal dihitung dari `df.iloc[-2]` (candle terakhir yang close), bukan `iloc[-1]` (forming)
- Backtest trades diberi `source='backtest'`, live trades `source='live'` — statistik /stats hanya tampilkan live
- ML default predict = `(None, 0.0)` — bot tidak trade jika model belum siap (fail-safe)
- Model dilindungi `model_lock` (RLock) mencegah race condition predict vs retrain
- `chat_ids.json` ditulis dengan atomic write (tempfile + os.replace)
- `trend_bull` (SMA200) dan `sar_bull` (Parabolic SAR) disimpan ke DB dan dipakai sebagai fitur ML
- TIMEOUT outcome: trade force-close setelah MAX_TRADE_CANDLES (60) — dicatat di DB, tidak dihitung WIN/LOSE

**Why:** Race condition model/predict bisa crash intermiten; default BUY berbahaya saat error.

## Signal Filter (urutan)
1. Ensemble ratio >= MIN_ENSEMBLE_RATIO (0.70) dari total votes aktif
2. Konfirmasi arah: SELL pakai SMA200 gate (terbukti WR 43.8%); BUY pakai RSI oversold + MACD positif
3. ATR filter: skip jika ATR < ATR_MIN_THRESHOLD (0.5) — pasar terlalu flat
4. ML konfirmasi: ml_proba >= ML_PROBA_THRESHOLD (0.58)

## Backtest Reset & Holdout Policy
- `reset_backtest_trades()` hanya hapus `source='backtest'` — JANGAN tambah `OR source IS NULL`
- `init_db()` menormalisasi NULL → 'live' sekali saat startup sebelum reset bisa dipanggil
- Reset dipanggil di `main.on_data_ready()` setiap startup → backtest selalu fresh dari data terbaru
- BACKTEST_HOLDOUT = 500: candle terakhir dikecualikan dari backtest training (cegah data leakage)
- Deriv API max 5000 candle @5-menit (~25 hari kalender / ~18 hari trading)

## Training Policy
- live < 200: backtest + live×5 (warmup backtest, live lebih berbobot)
- live >= 200: live saja (cukup data nyata, buang backtest)
- Threshold diatur via `_LIVE_ONLY_THRESHOLD = 200` di database.py
- TIMEOUT trades masuk DB tapi di-label 0 (LOSE) oleh ML retrain — konservatif tapi aman

## Retrain Adaptif (Fix 4)
- < 50 live trade: retrain setiap 10 trade (adaptasi cepat)
- 50–199 live trade: retrain setiap 25 trade
- ≥ 200 live trade: retrain setiap 50 trade (stabil)
- Helper: `_get_retrain_interval(live_count)` di signal_generator.py

## DB Connection (Fix 8)
- Thread-local persistent connection via `threading.local()` di database.py
- Tidak dibuka/tutup tiap query — reuse per thread selama bot berjalan
- `_reset_connection()` membersihkan koneksi rusak (rollback + close + del)
- PRAGMA synchronous=NORMAL + WAL = performa lebih cepat, tetap aman

## Label Simulasi (Fix 2)
- SPREAD_ESTIMATE = $0.50 diterapkan ke cek TP (bukan SL)
- BUY WIN: highs[j] >= tp + SPREAD_ESTIMATE (TP lebih sulit dicapai → label lebih realistis)
- SELL WIN: lows[j] <= tp - SPREAD_ESTIMATE

## Indikator Inkremental (Fix 7)
- INDICATOR_WINDOW = 600 candle — hanya window terakhir yang dihitung ulang saat candle baru
- 600 candle cukup untuk semua indikator period ≤ 200 + buffer konvergensi EWM
- initial_train tetap pakai full dataframe (untuk backtest coverage)

## Config Akurasi
- TP = 2.5×ATR, SL = 1.0×ATR → R:R = 2.5:1
- RF: 150 trees, max_depth=8, class_weight='balanced'
- SIGNAL_CHECK_INTERVAL = 10s
- Feature importance disimpan ke feature_importance.json setiap retrain (Fix 6)
