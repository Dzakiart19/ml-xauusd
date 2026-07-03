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
- `Retrain` — auto-retrain setiap RETRAIN_EVERY live trade selesai

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
- `trend_bull` (SMA200 filter) disimpan ke DB dan dipakai sebagai fitur ML

**Why:** Race condition model/predict bisa crash intermiten; default BUY berbahaya saat error.

## Signal Filter (urutan)
1. Ensemble ratio >= MIN_ENSEMBLE_RATIO (0.65) dari total votes aktif
2. Trend filter: BUY hanya jika close > SMA200, SELL hanya jika close < SMA200
3. ATR filter: skip jika ATR < ATR_MIN_THRESHOLD (0.5) — pasar terlalu flat
4. ML konfirmasi: ml_proba >= ML_PROBA_THRESHOLD (0.58)

## Backtest Reset Policy
- `reset_backtest_trades()` hanya hapus `source='backtest'` — JANGAN tambah `OR source IS NULL`
- `init_db()` menormalisasi NULL → 'live' sekali saat startup sebelum reset bisa dipanggil
- Reset dipanggil di `main.on_data_ready()` setiap startup → backtest selalu fresh dari data terbaru
- Deriv API memberikan max ~2710 candle @5-menit (~9.4 hari), bukan 3500 penuh (weekend/market close)

## Training Policy
- live < 200: backtest + live×5 (warmup backtest, live lebih berbobot)
- live >= 200: live saja (cukup data nyata, buang backtest)
- Threshold diatur via `_LIVE_ONLY_THRESHOLD = 200` di database.py

## Config Akurasi
- TP = 2.5×ATR, SL = 1.0×ATR → R:R = 2.5:1
- RF: 150 trees, max_depth=8
- SIGNAL_CHECK_INTERVAL = 10s (naik dari 5s untuk kurangi APScheduler warning)
