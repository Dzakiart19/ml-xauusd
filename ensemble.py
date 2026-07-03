"""
Shared ensemble voting logic — dipakai oleh SignalGenerator dan backtest.
Mengembalikan (bull, bear, total_votes) agar skor bisa dinormalisasi.
"""

import numpy as np
import pandas as pd


def safe_get(row, key, default=np.nan):
    """Ambil nilai dari row (dict/Series), kembalikan default jika NaN."""
    val = row.get(key, default)
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


def ensemble_vote(row: pd.Series) -> tuple:
    """
    Voting 10 indikator. Indikator kondisional (RSI, Stoch, BB) hanya
    ikut vote jika di zona ekstrem — sehingga total_votes bisa < 10.

    Return: (bull_score, bear_score, total_votes_cast)
    """
    bull, bear, total = 0, 0, 0

    def valid(val):
        return not (isinstance(val, float) and np.isnan(val))

    # 1. EMA10 vs EMA21 — selalu vote
    total += 1
    if row.get("ema_cross", 0) == 1:
        bull += 1
    else:
        bear += 1

    # 2. EMA21 vs EMA50
    e21 = row.get("EMA_21", np.nan)
    e50 = row.get("EMA_50", np.nan)
    if valid(e21) and valid(e50):
        total += 1
        if e21 > e50:
            bull += 1
        else:
            bear += 1

    # 3. Harga vs SMA50
    sma50 = row.get("SMA_50", np.nan)
    close = row.get("close",  np.nan)
    if valid(sma50) and valid(close):
        total += 1
        if close > sma50:
            bull += 1
        else:
            bear += 1

    # 4. MACD histogram
    macdh = row.get("MACDh_12_26_9", np.nan)
    if valid(macdh):
        total += 1
        if macdh > 0:
            bull += 1
        else:
            bear += 1

    # 5. RSI — hanya vote di zona oversold/overbought
    rsi = row.get("RSI_14", np.nan)
    if valid(rsi) and (rsi < 40 or rsi > 60):
        total += 1
        if rsi < 40:
            bull += 1
        else:
            bear += 1

    # 6. Stochastic — hanya vote di zona ekstrem
    stoch_k = row.get("STOCHk_14_3_3", np.nan)
    if valid(stoch_k) and (stoch_k < 25 or stoch_k > 75):
        total += 1
        if stoch_k < 25:
            bull += 1
        else:
            bear += 1

    # 7. Bollinger Band position — hanya vote di zona ekstrem
    bb = row.get("bb_pos", np.nan)
    if valid(bb) and (bb < 0.2 or bb > 0.8):
        total += 1
        if bb < 0.2:
            bull += 1
        else:
            bear += 1

    # 8. Parabolic SAR — selalu vote
    total += 1
    if row.get("sar_bull", 0) == 1:
        bull += 1
    else:
        bear += 1

    # 9. Pola candlestick bullish
    if row.get("bullish_cdl", 0) > 0:
        total += 1
        bull += 1

    # 10. Pola candlestick bearish
    if row.get("bearish_cdl", 0) > 0:
        total += 1
        bear += 1

    return bull, bear, max(total, 1)
