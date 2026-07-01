"""
Generator sinyal trading XAUUSD.
- Loop setiap SIGNAL_CHECK_INTERVAL detik
- Hitung ulang indikator ketika ada candle baru (candles_dirty flag)
- Lacak sinyal aktif (TP/SL) via tick real-time
- Auto-retrain setiap RETRAIN_EVERY trade selesai
"""

import logging
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    ATR_TP_MULTIPLIER, ATR_SL_MULTIPLIER,
    MIN_ENSEMBLE_SCORE, SIGNAL_CHECK_INTERVAL,
    RETRAIN_EVERY,
)
from indicators import calculate_indicators, extract_features
from ml_model import XAUModel
from database import (
    log_trade, update_trade_outcome,
    count_completed_trades, load_all_trades_for_training,
    log_evaluation,
)
from backtest import run_backtest

logger = logging.getLogger(__name__)


class SignalGenerator:
    def __init__(self, shared_state: dict, state_lock: threading.Lock,
                 send_message_fn):
        self.shared_state    = shared_state
        self.state_lock      = state_lock
        self.send_message    = send_message_fn

        self.model           = XAUModel()
        self._stop_evt       = threading.Event()
        self._started        = threading.Event()   # guard: start() sekali saja
        self._thread         = None

        # Trade aktif
        self._active_trade   = None
        self._trade_db_id    = None

    # ─── Public ───────────────────────────────────────────────────────────────

    def initial_train(self):
        """Dipanggil satu kali setelah data historis siap."""
        with self.state_lock:
            df = self.shared_state.get("candles", pd.DataFrame())

        if df.empty:
            logger.warning("Tidak ada candle untuk pelatihan awal.")
            return

        df_ind = calculate_indicators(df)

        with self.state_lock:
            self.shared_state["candles_with_indicators"] = df_ind
            self.shared_state["candles_dirty"]           = False
            self.shared_state["is_ready"]                = True

        # ── Backtest historis (hanya jika database kosong) ────────────────
        bt_count = run_backtest(df_ind)

        # ── Latih model: pakai data backtest jika cukup, fallback ke simulasi
        trades      = load_all_trades_for_training()
        use_backtest = bt_count > 0 and len(trades) >= 20

        if use_backtest:
            acc  = self.model.retrain(trades)
            wins   = sum(1 for t in trades if t["outcome"] == "WIN")
            losses = len(trades) - wins
            wr     = wins / len(trades) * 100

            self.send_message(
                f"🤖 *Bot XAUUSD Signal Aktif!*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Data historis  : `{len(df)}` candle\n"
                f"🔁 Backtest       : `{bt_count}` sinyal\n"
                f"✅ WIN `{wins}` | ❌ LOSE `{losses}`\n"
                f"📈 Win rate       : `{wr:.1f}%`\n"
                f"⏱ Timeframe      : `5 Menit`\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Model dilatih dari data historis\\.\n"
                f"/stats untuk statistik | /ping untuk status"
            )
        else:
            # Data backtest kurang dari 20 — pakai simulasi label biasa
            acc = self.model.initial_train(df_ind)
            extra = f" \\(backtest: `{bt_count}` sinyal\\)" if bt_count > 0 else ""
            self.send_message(
                f"🤖 *Bot XAUUSD Signal Aktif!*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Data historis : `{len(df)}` candle{extra}\n"
                f"🎯 Akurasi CV    : `{acc:.1%}`\n"
                f"⏱ Timeframe     : `5 Menit`\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Gunakan /stats untuk statistik\n"
                f"/ping untuk cek status bot"
            )

        logger.info("Model awal siap.")

    def start(self):
        """Mulai loop — hanya bisa dipanggil sekali."""
        if self._started.is_set():
            logger.warning("SignalGenerator sudah berjalan, start() diabaikan.")
            return
        self._started.set()
        self._thread = threading.Thread(
            target=self._loop, name="SignalGen", daemon=True
        )
        self._thread.start()
        logger.info("SignalGenerator thread dimulai.")

    def stop(self):
        self._stop_evt.set()

    # ─── Loop utama ───────────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop_evt.is_set():
            try:
                with self.state_lock:
                    is_ready = self.shared_state.get("is_ready", False)
                    dirty    = self.shared_state.get("candles_dirty", False)

                if not is_ready:
                    time.sleep(SIGNAL_CHECK_INTERVAL)
                    continue

                # ── Hitung ulang indikator jika candle baru datang ─────────
                if dirty:
                    self._recalculate_indicators()

                # ── Lacak trade aktif ──────────────────────────────────────
                if self._active_trade:
                    self._track_active_trade()
                else:
                    self._try_generate_signal()

            except Exception as e:
                logger.error(f"Error di loop: {e}", exc_info=True)

            time.sleep(SIGNAL_CHECK_INTERVAL)

    # ─── Update indikator ─────────────────────────────────────────────────────

    def _recalculate_indicators(self):
        with self.state_lock:
            df = self.shared_state.get("candles")
            if df is None or df.empty:
                return
            df_copy = df.copy()

        try:
            df_ind = calculate_indicators(df_copy)
            with self.state_lock:
                self.shared_state["candles_with_indicators"] = df_ind
                self.shared_state["candles_dirty"]           = False
        except Exception as e:
            logger.error(f"Gagal hitung ulang indikator: {e}", exc_info=True)

    # ─── Lacak trade aktif ─────────────────────────────────────────────────────

    def _track_active_trade(self):
        with self.state_lock:
            price = self.shared_state.get("current_price")

        if price is None:
            return

        trade     = self._active_trade
        direction = trade["direction"]
        entry     = trade["entry_price"]
        tp        = trade["tp"]
        sl        = trade["sl"]

        hit = None
        if direction == "BUY":
            if price >= tp:  hit = "WIN"
            elif price <= sl: hit = "LOSE"
        else:
            if price <= tp:  hit = "WIN"
            elif price >= sl: hit = "LOSE"

        if not hit:
            return

        pips  = abs(price - entry) / 0.1
        sign  = "+" if hit == "WIN" else "-"
        emoji = "✅" if hit == "WIN" else "❌"

        self.send_message(
            f"{emoji} *{hit}* — Trade Selesai!\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Arah   : `{direction}` XAUUSD\n"
            f"Entry  : `{entry:.2f}`\n"
            f"Tutup  : `{price:.2f}`\n"
            f"Pips   : `{sign}{abs(pips):.1f}`"
        )

        if self._trade_db_id:
            update_trade_outcome(self._trade_db_id, hit, pips)

        self._active_trade = None
        self._trade_db_id  = None

        # Cek retrain
        completed = count_completed_trades()
        if completed > 0 and completed % RETRAIN_EVERY == 0:
            threading.Thread(
                target=self._auto_retrain, daemon=True, name="Retrain"
            ).start()

    # ─── Generate sinyal ──────────────────────────────────────────────────────

    def _try_generate_signal(self):
        with self.state_lock:
            df    = self.shared_state.get("candles_with_indicators")
            price = self.shared_state.get("current_price")

        if df is None or df.empty or price is None:
            return

        last = df.iloc[-1]

        # Ensemble voting
        bull_score, bear_score = self._ensemble_vote(last)

        # ML prediction
        features          = extract_features(df)
        ml_label, ml_proba = self.model.predict(features.iloc[-1])
        ml_bull = (ml_label == 1)
        ml_bear = (ml_label == 0)

        direction = None
        if bull_score >= MIN_ENSEMBLE_SCORE and ml_bull:
            direction = "BUY"
        elif bear_score >= MIN_ENSEMBLE_SCORE and ml_bear:
            direction = "SELL"

        if direction is None:
            return

        # ATR untuk TP/SL
        atr = float(last.get("ATRr_14", 1.0))
        if np.isnan(atr) or atr <= 0:
            atr = 1.0

        entry = price
        if direction == "BUY":
            tp = entry + ATR_TP_MULTIPLIER * atr
            sl = entry - ATR_SL_MULTIPLIER * atr
        else:
            tp = entry - ATR_TP_MULTIPLIER * atr
            sl = entry + ATR_SL_MULTIPLIER * atr

        score_used = bull_score if direction == "BUY" else bear_score
        arrow      = "🟢" if direction == "BUY" else "🔴"

        self.send_message(
            f"{arrow} *SINYAL {direction} — XAUUSD*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Entry : `{entry:.2f}`\n"
            f"🎯 TP    : `{tp:.2f}`  (+{ATR_TP_MULTIPLIER}×ATR)\n"
            f"🛑 SL    : `{sl:.2f}`  (-{ATR_SL_MULTIPLIER}×ATR)\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Ensemble: `{score_used}/10` | ML: `{ml_proba:.0%}`\n"
            f"📏 ATR: `{atr:.2f}` | ⏰ `{datetime.now(timezone.utc).strftime('%H:%M UTC')}`"
        )

        trade_data = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "direction":      direction,
            "entry_price":    entry,
            "tp":             tp,
            "sl":             sl,
            "rsi":            self._safe(last, "RSI_14"),
            "macd_hist":      self._safe(last, "MACDh_12_26_9"),
            "atr":            atr,
            "bb_pos":         self._safe(last, "bb_pos"),
            "ema_signal":     int(self._safe(last, "ema_cross", 0)),
            "stoch_k":        self._safe(last, "STOCHk_14_3_3"),
            "stoch_d":        self._safe(last, "STOCHd_14_3_3"),
            "cci":            self._safe(last, "CCI_20_0.015"),
            "willr":          self._safe(last, "WILLR_14"),
            "mfi":            self._safe(last, "MFI_14"),
            "bullish_cdl":    int(self._safe(last, "bullish_cdl", 0)),
            "bearish_cdl":    int(self._safe(last, "bearish_cdl", 0)),
            "ensemble_score": score_used,
            "ml_proba":       ml_proba,
        }
        trade_id = log_trade(trade_data)

        self._active_trade = {
            "direction":   direction,
            "entry_price": entry,
            "tp":          tp,
            "sl":          sl,
        }
        self._trade_db_id = trade_id

        logger.info(
            f"Sinyal {direction}: Entry={entry:.2f} TP={tp:.2f} SL={sl:.2f}"
        )

    # ─── Ensemble voting (0–10 per side) ─────────────────────────────────────

    def _ensemble_vote(self, row: pd.Series) -> tuple:
        bull, bear = 0, 0

        def v(val):
            return not (isinstance(val, float) and np.isnan(val))

        # 1. EMA10 vs EMA21
        if row.get("ema_cross", 0) == 1: bull += 1
        else:                             bear += 1

        # 2. EMA21 vs EMA50
        e21, e50 = row.get("EMA_21", np.nan), row.get("EMA_50", np.nan)
        if v(e21) and v(e50):
            if e21 > e50: bull += 1
            else:         bear += 1

        # 3. Harga vs SMA50
        sma50 = row.get("SMA_50", np.nan)
        close = row.get("close",  np.nan)
        if v(sma50) and v(close):
            if close > sma50: bull += 1
            else:             bear += 1

        # 4. MACD histogram
        macdh = row.get("MACDh_12_26_9", np.nan)
        if v(macdh):
            if macdh > 0: bull += 1
            else:         bear += 1

        # 5. RSI
        rsi = row.get("RSI_14", 50)
        if v(rsi):
            if rsi < 40:   bull += 1
            elif rsi > 60: bear += 1

        # 6. Stochastic
        stoch_k = row.get("STOCHk_14_3_3", 50)
        if v(stoch_k):
            if stoch_k < 25:   bull += 1
            elif stoch_k > 75: bear += 1

        # 7. Bollinger Band position
        bb = row.get("bb_pos", 0.5)
        if v(bb):
            if bb < 0.2:   bull += 1
            elif bb > 0.8: bear += 1

        # 8. Parabolic SAR
        sar_bull = row.get("sar_bull", 0)
        if sar_bull == 1: bull += 1
        else:             bear += 1

        # 9. Candlestick patterns
        if row.get("bullish_cdl", 0) > 0: bull += 1
        if row.get("bearish_cdl", 0) > 0: bear += 1

        return bull, bear

    # ─── Auto retrain ─────────────────────────────────────────────────────────

    def _auto_retrain(self):
        logger.info("Auto-retrain dimulai...")
        trades = load_all_trades_for_training()
        if len(trades) < 20:
            return

        wins   = sum(1 for t in trades if t["outcome"] == "WIN")
        losses = sum(1 for t in trades if t["outcome"] == "LOSE")
        total  = len(trades)

        self.model.retrain(trades)
        actual_wr = wins / total * 100
        log_evaluation(actual_wr, total, wins, losses)

        self.send_message(
            f"🔄 *Auto-Evaluasi & Retrain Selesai*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Total  : `{total}` trade\n"
            f"✅ Menang : `{wins}`\n"
            f"❌ Kalah  : `{losses}`\n"
            f"📈 Win Rate: `{actual_wr:.1f}%`\n"
            f"🤖 Model dilatih ulang & disimpan ke disk"
        )

    @staticmethod
    def _safe(row, key, default=np.nan):
        val = row.get(key, default)
        if isinstance(val, float) and np.isnan(val):
            return default
        return val
