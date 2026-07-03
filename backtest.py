"""
Backtest historis — replay candle historis untuk mengisi database
dengan sinyal simulasi sebelum bot mulai live trading.

Hanya berjalan sekali saat startup jika database masih kosong.
Backtest menggunakan ensemble vote saja (tanpa ML gate) karena model
belum ada saat startup — trade backtest diberi label source='backtest'
agar statistik live tidak tercampur.
"""

import logging
from datetime import timezone

import numpy as np
import pandas as pd

from config import (
    ATR_TP_MULTIPLIER, ATR_SL_MULTIPLIER,
    MIN_ENSEMBLE_RATIO, LABEL_LOOKAHEAD, ATR_MIN_THRESHOLD,
    BUY_RSI_MAX, SELL_RSI_MIN, SELL_STOCH_MIN,
    BACKTEST_HOLDOUT, SPREAD_ESTIMATE,
)
from database import log_trade, update_trade_outcome, count_backtest_trades
from ensemble import ensemble_vote, safe_get

logger = logging.getLogger(__name__)


# ─── Simulasi outcome ─────────────────────────────────────────────────────────

def _simulate_outcome(df: pd.DataFrame, entry_idx: int,
                      direction: str, tp: float, sl: float) -> tuple:
    """
    Lihat candle ke depan (max LABEL_LOOKAHEAD).
    Return (outcome, pips, exit_idx).
    outcome: 'WIN', 'LOSE', atau None jika tidak ada resolusi.
    """
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    entry  = closes[entry_idx]

    end_idx = min(entry_idx + 1 + LABEL_LOOKAHEAD, len(df))
    for j in range(entry_idx + 1, end_idx):
        if direction == "BUY":
            # Fix 2: TP harus diraih di atas spread (entry sesungguhnya = ask = mid + spread)
            if highs[j] >= tp + SPREAD_ESTIMATE:
                return "WIN",  abs(tp    - entry) / 0.01, j
            if lows[j]  <= sl:
                return "LOSE", abs(entry - sl)    / 0.01, j
        else:  # SELL
            # Fix 2: TP harus diraih di bawah spread (entry sesungguhnya = bid = mid - spread)
            if lows[j]  <= tp - SPREAD_ESTIMATE:
                return "WIN",  abs(entry - tp)    / 0.01, j
            if highs[j] >= sl:
                return "LOSE", abs(sl    - entry) / 0.01, j

    return None, 0.0, end_idx


# ─── Entry point utama ────────────────────────────────────────────────────────

def run_backtest(df_ind: pd.DataFrame) -> int:
    """
    Jalankan backtest pada df_ind (DataFrame dengan indikator sudah dihitung).
    Hanya berjalan jika database masih kosong.
    Return jumlah trade yang dihasilkan.
    """
    if count_backtest_trades() > 0:
        logger.info("Data backtest sudah ada — backtest dilewati.")
        return 0

    logger.info("Memulai backtest historis...")

    closes = df_ind['close'].values
    times  = df_ind.index
    n      = len(df_ind)

    # Mulai dari candle ke-210 supaya SMA200 + indikator sudah stabil
    # end_idx dikurangi BACKTEST_HOLDOUT agar candle terbaru tidak dipakai untuk training
    # (Fix 1: cegah data leakage — candle terakhir reserved untuk sinyal live awal)
    start_idx    = min(210, n // 3)
    end_idx      = n - max(LABEL_LOOKAHEAD, BACKTEST_HOLDOUT)
    signal_count = 0
    active_until = -1
    _results     = []   # track outcome setiap trade untuk laporan akurat

    for i in range(start_idx, end_idx):
        if i <= active_until:
            continue

        row  = df_ind.iloc[i]
        bull, bear, total = ensemble_vote(row)

        # SMA200 disimpan sebagai fitur ML (bukan gate arah)
        trend_bull = int(safe_get(row, "trend_bull", 0))

        # Konfirmasi arah: RSI + MACD + Stoch (data-driven, bukan SMA200)
        rsi     = float(safe_get(row, "RSI_14",         50.0))
        macd_h  = float(safe_get(row, "MACDh_12_26_9",   0.0))
        stoch_k = float(safe_get(row, "STOCHk_14_3_3",  50.0))

        direction = None
        score     = 0
        # SELL: hanya saat harga di bawah SMA200 (terbukti WR 43.8%) + proteksi RSI/stoch
        if bear / total >= MIN_ENSEMBLE_RATIO and trend_bull == 0 \
                and rsi > SELL_RSI_MIN and stoch_k > SELL_STOCH_MIN:
            direction, score = "SELL", bear
        # BUY: RSI harus benar-benar oversold (< 45) + MACD momentum positif
        elif bull / total >= MIN_ENSEMBLE_RATIO and rsi < BUY_RSI_MAX and macd_h > 0:
            direction, score = "BUY", bull

        if direction is None:
            continue

        entry = float(closes[i])
        atr   = float(safe_get(row, "ATRr_14", 1.0))
        if np.isnan(atr) or atr <= 0:
            atr = 1.0

        # Filter ATR minimum — jangan trade saat pasar terlalu flat
        if atr < ATR_MIN_THRESHOLD:
            continue

        if direction == "BUY":
            tp = entry + ATR_TP_MULTIPLIER * atr
            sl = entry - ATR_SL_MULTIPLIER * atr
        else:
            tp = entry - ATR_TP_MULTIPLIER * atr
            sl = entry + ATR_SL_MULTIPLIER * atr

        outcome, pips, exit_idx = _simulate_outcome(df_ind, i, direction, tp, sl)
        if outcome is None:
            continue

        trade_data = {
            "timestamp":       times[i].isoformat(),
            "direction":       direction,
            "entry_price":     entry,
            "tp":              tp,
            "sl":              sl,
            "rsi":             safe_get(row, "RSI_14"),
            "macd_hist":       safe_get(row, "MACDh_12_26_9"),
            "atr":             atr,
            "bb_pos":          safe_get(row, "bb_pos"),
            "ema_signal":      int(safe_get(row, "ema_cross", 0)),
            "stoch_k":         safe_get(row, "STOCHk_14_3_3"),
            "stoch_d":         safe_get(row, "STOCHd_14_3_3"),
            "cci":             safe_get(row, "CCI_20_0.015"),
            "willr":           safe_get(row, "WILLR_14"),
            "mfi":             safe_get(row, "MFI_14", np.nan),
            "bullish_cdl":     int(safe_get(row, "bullish_cdl", 0)),
            "bearish_cdl":     int(safe_get(row, "bearish_cdl", 0)),
            "ensemble_score":  score,
            "ml_proba":        0.5,   # tidak ada ML saat backtest
            "source":          "backtest",
            "trend_bull":      trend_bull,
            "sar_bull":        int(safe_get(row, "sar_bull", 0)),
        }

        trade_id = log_trade(trade_data)
        update_trade_outcome(trade_id, outcome, pips)
        _results.append(outcome)
        signal_count += 1
        active_until = exit_idx

    if signal_count > 0:
        wins   = sum(1 for r in _results if r == "WIN")
        losses = sum(1 for r in _results if r == "LOSE")
        wr     = wins / signal_count * 100
        logger.info(
            f"Backtest selesai: {signal_count} sinyal | "
            f"WIN {wins} | LOSE {losses} | Win rate: {wr:.1f}%"
        )
    else:
        logger.info("Backtest selesai: tidak ada sinyal yang memenuhi syarat.")

    return signal_count
