"""
Backtest historis — replay candle historis untuk mengisi database
dengan sinyal simulasi sebelum bot mulai live trading.

Hanya berjalan sekali saat startup jika database masih kosong.
"""

import logging
from datetime import timezone

import numpy as np
import pandas as pd

from config import (
    ATR_TP_MULTIPLIER, ATR_SL_MULTIPLIER,
    MIN_ENSEMBLE_SCORE, LABEL_LOOKAHEAD,
)
from database import log_trade, update_trade_outcome, count_completed_trades

logger = logging.getLogger(__name__)


# ─── Ensemble vote (sama persis dengan SignalGenerator) ───────────────────────

def _ensemble_vote(row: pd.Series):
    bull, bear = 0, 0

    def v(val):
        return not (isinstance(val, float) and np.isnan(val))

    if row.get("ema_cross", 0) == 1: bull += 1
    else:                             bear += 1

    e21, e50 = row.get("EMA_21", np.nan), row.get("EMA_50", np.nan)
    if v(e21) and v(e50):
        if e21 > e50: bull += 1
        else:         bear += 1

    sma50 = row.get("SMA_50", np.nan)
    close = row.get("close",  np.nan)
    if v(sma50) and v(close):
        if close > sma50: bull += 1
        else:             bear += 1

    macdh = row.get("MACDh_12_26_9", np.nan)
    if v(macdh):
        if macdh > 0: bull += 1
        else:         bear += 1

    rsi = row.get("RSI_14", 50)
    if v(rsi):
        if rsi < 40:   bull += 1
        elif rsi > 60: bear += 1

    stoch_k = row.get("STOCHk_14_3_3", 50)
    if v(stoch_k):
        if stoch_k < 25:   bull += 1
        elif stoch_k > 75: bear += 1

    bb = row.get("bb_pos", 0.5)
    if v(bb):
        if bb < 0.2:   bull += 1
        elif bb > 0.8: bear += 1

    sar_bull = row.get("sar_bull", 0)
    if sar_bull == 1: bull += 1
    else:             bear += 1

    if row.get("bullish_cdl", 0) > 0: bull += 1
    if row.get("bearish_cdl", 0) > 0: bear += 1

    return bull, bear


def _safe(row, key, default=np.nan):
    val = row.get(key, default)
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


# ─── Simulasi outcome: cek apakah TP atau SL tercapai di candle berikutnya ────

def _simulate_outcome(df: pd.DataFrame, entry_idx: int,
                      direction: str, tp: float, sl: float) -> tuple:
    """
    Lihat candle ke depan (max LABEL_LOOKAHEAD).
    Return (outcome, pips, exit_idx) — exit_idx adalah candle tempat trade selesai.
    outcome bisa 'WIN', 'LOSE', atau None jika tidak ada resolusi.
    """
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    entry  = closes[entry_idx]

    end_idx = min(entry_idx + 1 + LABEL_LOOKAHEAD, len(df))
    for j in range(entry_idx + 1, end_idx):
        if direction == "BUY":
            if highs[j] >= tp:
                return "WIN",  abs(tp    - entry) / 0.1, j
            if lows[j]  <= sl:
                return "LOSE", abs(entry - sl)    / 0.1, j
        else:  # SELL
            if lows[j]  <= tp:
                return "WIN",  abs(entry - tp)    / 0.1, j
            if highs[j] >= sl:
                return "LOSE", abs(sl    - entry) / 0.1, j

    return None, 0.0, end_idx


# ─── Entry point utama ────────────────────────────────────────────────────────

def run_backtest(df_ind: pd.DataFrame) -> int:
    """
    Jalankan backtest pada df_ind (DataFrame dengan indikator sudah dihitung).
    Hanya berjalan jika database masih kosong.
    Return jumlah trade yang dihasilkan.
    """
    if count_completed_trades() > 0:
        logger.info("Database sudah ada data — backtest dilewati.")
        return 0

    logger.info("Memulai backtest historis...")

    closes    = df_ind['close'].values
    highs     = df_ind['high'].values
    lows      = df_ind['low'].values
    times     = df_ind.index

    n         = len(df_ind)
    # Mulai dari candle ke-60 supaya indikator sudah stabil
    start_idx = min(60, n // 4)
    # Sisakan LABEL_LOOKAHEAD candle di akhir untuk simulasi outcome
    end_idx   = n - LABEL_LOOKAHEAD

    signal_count = 0
    active_until = -1   # index candle sampai trade aktif selesai

    for i in range(start_idx, end_idx):
        if i <= active_until:
            continue   # masih ada trade aktif, skip

        row  = df_ind.iloc[i]
        bull, bear = _ensemble_vote(row)

        direction = None
        score     = 0
        if bull >= MIN_ENSEMBLE_SCORE:
            direction, score = "BUY",  bull
        elif bear >= MIN_ENSEMBLE_SCORE:
            direction, score = "SELL", bear

        if direction is None:
            continue

        entry = float(closes[i])
        atr   = float(_safe(row, "ATRr_14", 1.0))
        if np.isnan(atr) or atr <= 0:
            atr = 1.0

        if direction == "BUY":
            tp = entry + ATR_TP_MULTIPLIER * atr
            sl = entry - ATR_SL_MULTIPLIER * atr
        else:
            tp = entry - ATR_TP_MULTIPLIER * atr
            sl = entry + ATR_SL_MULTIPLIER * atr

        outcome, pips, exit_idx = _simulate_outcome(df_ind, i, direction, tp, sl)
        if outcome is None:
            continue   # tidak ada TP/SL dalam window — skip

        trade_data = {
            "timestamp":       times[i].isoformat(),
            "direction":       direction,
            "entry_price":     entry,
            "tp":              tp,
            "sl":              sl,
            "rsi":             _safe(row, "RSI_14"),
            "macd_hist":       _safe(row, "MACDh_12_26_9"),
            "atr":             atr,
            "bb_pos":          _safe(row, "bb_pos"),
            "ema_signal":      int(_safe(row, "ema_cross", 0)),
            "stoch_k":         _safe(row, "STOCHk_14_3_3"),
            "stoch_d":         _safe(row, "STOCHd_14_3_3"),
            "cci":             _safe(row, "CCI_20_0.015"),
            "willr":           _safe(row, "WILLR_14"),
            "mfi":             _safe(row, "MFI_14"),
            "bullish_cdl":     int(_safe(row, "bullish_cdl", 0)),
            "bearish_cdl":     int(_safe(row, "bearish_cdl", 0)),
            "ensemble_score":  score,
            "ml_proba":        0.5,   # belum ada prediksi ML saat backtest
        }

        trade_id = log_trade(trade_data)
        update_trade_outcome(trade_id, outcome, pips)
        signal_count += 1

        # Skip sampai trade ini selesai (bukan selalu 50 candle)
        active_until = exit_idx

    wins   = 0
    losses = 0
    if signal_count > 0:
        from database import load_all_trades_for_training
        trades = load_all_trades_for_training()
        wins   = sum(1 for t in trades if t["outcome"] == "WIN")
        losses = sum(1 for t in trades if t["outcome"] == "LOSE")
        wr     = wins / signal_count * 100

        logger.info(
            f"Backtest selesai: {signal_count} sinyal | "
            f"WIN {wins} | LOSE {losses} | Win rate: {wr:.1f}%"
        )
    else:
        logger.info("Backtest selesai: tidak ada sinyal yang memenuhi syarat.")

    return signal_count
