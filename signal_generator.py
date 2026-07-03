"""
Generator sinyal trading XAUUSD.
- Loop setiap SIGNAL_CHECK_INTERVAL detik
- Hitung ulang indikator HANYA saat candle baru close (candles_dirty flag)
- Lacak sinyal aktif (TP/SL) via tick real-time
- Auto-retrain setiap RETRAIN_EVERY trade live selesai
"""

import logging
import threading
import time
from datetime import datetime, timezone, timedelta

WIB = timezone(timedelta(hours=7))

import numpy as np
import pandas as pd

from config import (
    ATR_TP_MULTIPLIER, ATR_SL_MULTIPLIER, ATR_MIN_THRESHOLD,
    MIN_ENSEMBLE_RATIO, ML_PROBA_THRESHOLD,
    SIGNAL_CHECK_INTERVAL, RETRAIN_EVERY,
    BUY_RSI_MAX, SELL_RSI_MIN, SELL_STOCH_MIN,
)
from ensemble import ensemble_vote, safe_get
from indicators import calculate_indicators, extract_features
from ml_model import XAUModel
from database import (
    log_trade, update_trade_outcome,
    count_completed_live_trades, load_all_trades_for_training,
    log_evaluation,
)
from backtest import run_backtest
from bot import set_active_signal

logger = logging.getLogger(__name__)


class SignalGenerator:
    def __init__(self, shared_state: dict, state_lock: threading.Lock,
                 send_message_fn):
        self.shared_state = shared_state
        self.state_lock   = state_lock
        self.send_message = send_message_fn

        self.model        = XAUModel()
        self._stop_evt    = threading.Event()
        self._started     = threading.Event()
        self._thread      = None

        # Trade aktif
        self._active_trade = None
        self._trade_db_id  = None

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

        # ── Latih model ───────────────────────────────────────────────────
        trades       = load_all_trades_for_training()
        use_backtest = bt_count > 0 and len(trades) >= 20

        if use_backtest:
            acc  = self.model.retrain(trades)
            wins = sum(1 for t in trades if t["outcome"] == "WIN"
                       and t.get("source") == "backtest")
            losses = sum(1 for t in trades if t["outcome"] == "LOSE"
                         and t.get("source") == "backtest")
            wr   = wins / bt_count * 100 if bt_count > 0 else 0
            log_evaluation(wr, bt_count, wins, losses, source="backtest")

            self.send_message(
                f"🤖 *Bot XAUUSD Signal Aktif\\!*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Data historis  : `{len(df)}` candle\n"
                f"🔁 Backtest       : `{bt_count}` sinyal\n"
                f"✅ WIN `{wins}` \\| ❌ LOSE `{losses}`\n"
                f"📈 Win rate BT    : `{wr:.1f}%`\n"
                f"⏱ Timeframe      : `5 Menit`\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Model dilatih dari data backtest\\.\n"
                f"/stats untuk statistik live \\| /ping untuk status"
            )
        else:
            acc = self.model.initial_train(df_ind)
            extra = f" \\(backtest: `{bt_count}` sinyal\\)" if bt_count > 0 else ""
            self.send_message(
                f"🤖 *Bot XAUUSD Signal Aktif\\!*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Data historis : `{len(df)}` candle{extra}\n"
                f"🎯 Akurasi CV    : `{acc:.1%}`\n"
                f"⏱ Timeframe     : `5 Menit`\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Gunakan /stats untuk statistik live\n"
                f"/ping untuk cek status bot"
            )

        logger.info("Model awal siap.")

    def start(self):
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

                # Hitung ulang indikator HANYA saat ada candle baru close
                if dirty:
                    self._recalculate_indicators()

                # Lacak trade aktif atau coba generate sinyal baru
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

    # ─── Lacak trade aktif ────────────────────────────────────────────────────

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
            if price >= tp:   hit = "WIN"
            elif price <= sl: hit = "LOSE"
        else:
            if price <= tp:   hit = "WIN"
            elif price >= sl: hit = "LOSE"

        if not hit:
            return

        pips = abs(price - entry) / 0.01   # 1 pip XAUUSD = $0.01
        sign  = "+" if hit == "WIN" else "-"
        emoji = "✅" if hit == "WIN" else "❌"

        self.send_message(
            f"{emoji} *{hit}* — Trade Selesai\\!\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Arah   : `{direction}` XAUUSD\n"
            f"Entry  : `{entry:.2f}`\n"
            f"Tutup  : `{price:.2f}`\n"
            f"Pips   : `{sign}{pips:.0f}`"
        )

        if self._trade_db_id:
            update_trade_outcome(self._trade_db_id, hit, pips)

        self._active_trade = None
        self._trade_db_id  = None
        set_active_signal(None)

        # Trigger retrain berdasarkan trade LIVE saja
        live_done = count_completed_live_trades()
        if live_done > 0 and live_done % RETRAIN_EVERY == 0:
            threading.Thread(
                target=self._auto_retrain, daemon=True, name="Retrain"
            ).start()

    # ─── Generate sinyal ──────────────────────────────────────────────────────

    def _try_generate_signal(self):
        with self.state_lock:
            df    = self.shared_state.get("candles_with_indicators")
            price = self.shared_state.get("current_price")

        if df is None or len(df) < 2 or price is None:
            return

        # Gunakan candle TERAKHIR yang SUDAH CLOSE (bukan candle forming saat ini)
        # iloc[-1] = candle sedang terbentuk, iloc[-2] = candle yang baru close
        last  = df.iloc[-2]
        close = float(safe_get(last, "close", np.nan))

        # ── Ensemble voting ────────────────────────────────────────────────
        bull, bear, total = ensemble_vote(last)

        bull_ratio = bull / total
        bear_ratio = bear / total

        # ── SMA200: simpan sebagai fitur ML, bukan gate arah ─────────────
        sma200     = safe_get(last, "SMA_200", np.nan)
        trend_bull = 1 if (not np.isnan(sma200) and close > sma200) else 0

        # ── ATR filter: jangan trade pasar flat ───────────────────────────
        atr = float(safe_get(last, "ATRr_14", 0.0))
        if np.isnan(atr) or atr <= 0:
            atr = 1.0

        if atr < ATR_MIN_THRESHOLD:
            logger.debug(f"ATR terlalu kecil ({atr:.3f}) — skip sinyal.")
            return

        # ── Konfirmasi arah: RSI + MACD + Stoch (data-driven) ────────────
        # SMA200 dilepas sebagai gate karena bukti data: BUY di atas SMA200
        # memiliki WR 23.5% vs SELL di bawah SMA200 WR 43.8% → SMA200 gate
        # memperburuk BUY bukan memperbaikinya.
        rsi     = float(safe_get(last, "RSI_14",        50.0))
        macd_h  = float(safe_get(last, "MACDh_12_26_9",  0.0))
        stoch_k = float(safe_get(last, "STOCHk_14_3_3", 50.0))

        direction = None
        # SELL: hanya saat harga di bawah SMA200 (tren turun lokal) + RSI/stoch
        # aman — ini filter yang terbukti bagus (WR 43.8%)
        if bear_ratio >= MIN_ENSEMBLE_RATIO and trend_bull == 0 \
                and rsi > SELL_RSI_MIN and stoch_k > SELL_STOCH_MIN:
            direction = "SELL"
        # BUY: tanpa SMA200 gate (karena BUY di atas SMA200 justru buruk),
        # ganti dengan RSI oversold nyata + MACD momentum positif
        elif bull_ratio >= MIN_ENSEMBLE_RATIO and rsi < BUY_RSI_MAX and macd_h > 0:
            direction = "BUY"

        if direction is None:
            return

        # ── Konfirmasi ML — gunakan closed candle (iloc[-2]) ──────────────────
        features           = extract_features(df)
        ml_label, ml_proba = self.model.predict(features.iloc[-2])

        # Jika model belum siap (None) atau proba di bawah threshold → skip
        if ml_label is None:
            logger.debug("Model belum siap — sinyal ditunda.")
            return

        ml_confirms_buy  = (ml_label == 1 and ml_proba >= ML_PROBA_THRESHOLD)
        ml_confirms_sell = (ml_label == 0 and ml_proba >= ML_PROBA_THRESHOLD)

        if direction == "BUY"  and not ml_confirms_buy:
            return
        if direction == "SELL" and not ml_confirms_sell:
            return

        # ── Hitung TP / SL ─────────────────────────────────────────────────
        entry = price
        if direction == "BUY":
            tp = entry + ATR_TP_MULTIPLIER * atr
            sl = entry - ATR_SL_MULTIPLIER * atr
        else:
            tp = entry - ATR_TP_MULTIPLIER * atr
            sl = entry + ATR_SL_MULTIPLIER * atr

        score_used = bull if direction == "BUY" else bear
        arrow      = "🟢" if direction == "BUY" else "🔴"

        self.send_message(
            f"{arrow} *SINYAL {direction} — XAUUSD*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Entry : `{entry:.2f}`\n"
            f"🎯 TP    : `{tp:.2f}`  \\(\\+{ATR_TP_MULTIPLIER}×ATR\\)\n"
            f"🛑 SL    : `{sl:.2f}`  \\(\\-{ATR_SL_MULTIPLIER}×ATR\\)\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Ensemble: `{score_used}/{total}` votes \\| ML: `{ml_proba:.0%}`\n"
            f"📏 ATR: `{atr:.2f}` \\| 📈 Tren: `{'Bullish' if trend_bull else 'Bearish'}` \\| ⏰ `{datetime.now(WIB).strftime('%H:%M WIB')}`"
        )

        trade_data = {
            "timestamp":      datetime.now(WIB).isoformat(),
            "direction":      direction,
            "entry_price":    entry,
            "tp":             tp,
            "sl":             sl,
            "rsi":            safe_get(last, "RSI_14"),
            "macd_hist":      safe_get(last, "MACDh_12_26_9"),
            "atr":            atr,
            "bb_pos":         safe_get(last, "bb_pos"),
            "ema_signal":     int(safe_get(last, "ema_cross", 0)),
            "stoch_k":        safe_get(last, "STOCHk_14_3_3"),
            "stoch_d":        safe_get(last, "STOCHd_14_3_3"),
            "cci":            safe_get(last, "CCI_20_0.015"),
            "willr":          safe_get(last, "WILLR_14"),
            "mfi":            safe_get(last, "MFI_14"),
            "bullish_cdl":    int(safe_get(last, "bullish_cdl", 0)),
            "bearish_cdl":    int(safe_get(last, "bearish_cdl", 0)),
            "ensemble_score": score_used,
            "ml_proba":       ml_proba,
            "source":         "live",
            "trend_bull":     trend_bull,
        }
        trade_id = log_trade(trade_data)

        self._active_trade = {
            "direction":   direction,
            "entry_price": entry,
            "tp":          tp,
            "sl":          sl,
        }
        self._trade_db_id = trade_id
        set_active_signal({**self._active_trade, "timestamp": trade_data["timestamp"]})

        logger.info(
            f"Sinyal {direction}: Entry={entry:.2f} TP={tp:.2f} SL={sl:.2f} "
            f"Ensemble={score_used}/{total} ML={ml_proba:.0%}"
        )

    # ─── Auto retrain ─────────────────────────────────────────────────────────

    def _auto_retrain(self):
        logger.info("Auto-retrain dimulai...")
        trades = load_all_trades_for_training()
        if len(trades) < 20:
            return

        live_trades = [t for t in trades if t.get("source", "live") == "live"]
        wins   = sum(1 for t in live_trades if t["outcome"] == "WIN")
        losses = sum(1 for t in live_trades if t["outcome"] == "LOSE")
        total  = len(live_trades)

        self.model.retrain(trades)   # latih dengan semua data (live + backtest warmup)
        actual_wr = wins / total * 100 if total > 0 else 0
        log_evaluation(actual_wr, total, wins, losses, source="live")

        self.send_message(
            f"🔄 *Auto\\-Evaluasi & Retrain Selesai*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Trade live : `{total}`\n"
            f"✅ Menang     : `{wins}`\n"
            f"❌ Kalah      : `{losses}`\n"
            f"📈 Win Rate   : `{actual_wr:.1f}%`\n"
            f"🤖 Model dilatih ulang & disimpan"
        )
