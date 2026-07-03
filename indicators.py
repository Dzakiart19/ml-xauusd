"""
Kalkulasi indikator teknikal secara manual menggunakan pure pandas + numpy.
Tidak memerlukan library TA-Lib atau pandas-ta.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# INDIKATOR TEKNIKAL
# ══════════════════════════════════════════════════════════════════════════════

def calc_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(series: pd.Series,
              fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast    = calc_ema(series, fast)
    ema_slow    = calc_ema(series, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bbands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma   = series.rolling(window=period).mean()
    std   = series.rolling(window=period).std(ddof=0)
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def calc_stochastic(df: pd.DataFrame,
                    k_period: int = 14, d_period: int = 3) -> tuple:
    low_n  = df['low'].rolling(k_period).min()
    high_n = df['high'].rolling(k_period).max()
    denom  = (high_n - low_n).replace(0, np.nan)
    k = 100 * (df['close'] - low_n) / denom
    d = k.rolling(d_period).mean()
    return k, d


def calc_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp  = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return cci


def calc_willr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_n = df['high'].rolling(period).max()
    low_n  = df['low'].rolling(period).min()
    denom  = (high_n - low_n).replace(0, np.nan)
    willr  = -100 * (high_n - df['close']) / denom
    return willr


def calc_obv(df: pd.DataFrame) -> pd.Series:
    """OBV disertakan tapi TIDAK dipakai sebagai fitur ML (volume tidak riil)."""
    direction     = np.sign(df['close'].diff())
    direction.iloc[0] = 0
    obv = (direction * df['volume']).cumsum()
    return obv


def calc_stdev(series: pd.Series, period: int = 14) -> pd.Series:
    return series.rolling(period).std(ddof=0)


def calc_psar(df: pd.DataFrame,
              af_start: float = 0.02,
              af_step:  float = 0.02,
              af_max:   float = 0.20) -> tuple:
    """
    Kembalikan (psar, is_bull) sebagai pd.Series.
    is_bull = True berarti harga di atas SAR (uptrend).
    """
    high = df['high'].values
    low  = df['low'].values
    n    = len(df)

    psar_arr = np.full(n, np.nan)
    is_bull  = np.ones(n, dtype=bool)

    psar_arr[0] = low[0]
    hp = high[0]
    lp = low[0]
    af = af_start

    for i in range(1, n):
        prev_bull = is_bull[i - 1]

        if prev_bull:
            psar_i = psar_arr[i - 1] + af * (hp - psar_arr[i - 1])
            psar_i = min(psar_i, low[i - 1])
            if i >= 2:
                psar_i = min(psar_i, low[i - 2])

            if low[i] < psar_i:
                is_bull[i]  = False
                psar_arr[i] = hp
                lp = low[i]
                af = af_start
            else:
                is_bull[i]  = True
                psar_arr[i] = psar_i
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + af_step, af_max)
        else:
            psar_i = psar_arr[i - 1] - af * (psar_arr[i - 1] - lp)
            psar_i = max(psar_i, high[i - 1])
            if i >= 2:
                psar_i = max(psar_i, high[i - 2])

            if high[i] > psar_i:
                is_bull[i]  = True
                psar_arr[i] = lp
                hp = high[i]
                af = af_start
            else:
                is_bull[i]  = False
                psar_arr[i] = psar_i
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + af_step, af_max)

    idx = df.index
    return pd.Series(psar_arr, index=idx), pd.Series(is_bull, index=idx)


# ══════════════════════════════════════════════════════════════════════════════
# SMART MONEY INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def calc_fvg(df: pd.DataFrame, lookback: int = 30) -> tuple:
    """
    Fair Value Gap (FVG) — celah harga 3-candle yang belum terisi.

    Bullish FVG terbentuk di candle j jika: high[j-2] < low[j]
      → celah di bawah harga saat ini = zona demand/support
      → sinyal BUY jika harga turun ke dalam celah tsb

    Bearish FVG terbentuk di candle j jika: low[j-2] > high[j]
      → celah di atas harga saat ini = zona supply/resistance
      → sinyal SELL jika harga naik ke dalam celah tsb

    Return: (fvg_bull, fvg_bear) sebagai pd.Series int (0/1)
    """
    high  = df['high'].values
    low   = df['low'].values
    close = df['close'].values
    n     = len(df)

    fvg_bull = np.zeros(n, dtype=int)
    fvg_bear = np.zeros(n, dtype=int)

    for i in range(lookback, n):
        curr  = close[i]
        start = max(2, i - lookback)

        for j in range(start, i):
            # ── Bullish FVG: high[j-2] < low[j] ──────────────────────────────
            gap_bot = high[j - 2]
            gap_top = low[j]
            if gap_top > gap_bot and gap_bot * 0.999 <= curr <= gap_top * 1.005:
                fvg_bull[i] = 1
                break

            # ── Bearish FVG: low[j-2] > high[j] ──────────────────────────────
            gap_bot2 = high[j]
            gap_top2 = low[j - 2]
            if gap_top2 > gap_bot2 and gap_bot2 * 0.995 <= curr <= gap_top2 * 1.001:
                fvg_bear[i] = 1
                break

    idx = df.index
    return pd.Series(fvg_bull, index=idx), pd.Series(fvg_bear, index=idx)


def calc_fibonacci(df: pd.DataFrame,
                   lookback: int = 89,
                   tolerance: float = 0.002) -> tuple:
    """
    Fibonacci Retracement — deteksi harga mendekati level kunci (38.2%, 50%, 61.8%).

    Logika:
    - Temukan swing high/low dalam `lookback` candle terakhir
    - Tentukan arah swing: UP (low dulu → high) atau DOWN (high dulu → low)
    - Swing UP → level Fib = zona support = sinyal BUY
    - Swing DOWN → level Fib = zona resistance = sinyal SELL
    - tolerance = 0.2% dari harga (sekitar $8 untuk XAUUSD @$4000)

    Return: (fib_bull, fib_bear) sebagai pd.Series int (0/1)
    """
    high  = df['high'].values
    low   = df['low'].values
    close = df['close'].values
    n     = len(df)

    fib_bull = np.zeros(n, dtype=int)
    fib_bear = np.zeros(n, dtype=int)

    FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]

    for i in range(lookback, n):
        win      = slice(i - lookback, i + 1)
        win_h    = high[win]
        win_l    = low[win]

        idx_h    = int(np.argmax(win_h))
        idx_l    = int(np.argmin(win_l))
        s_high   = win_h[idx_h]
        s_low    = win_l[idx_l]
        rng      = s_high - s_low

        if rng < 1.0:          # pasar terlalu flat — skip
            continue

        curr = close[i]
        tol  = tolerance * curr  # toleransi dalam dollar

        if idx_l < idx_h:
            # Swing UP: low lebih dulu → high = retracement ke bawah = support
            for lvl in FIB_LEVELS:
                fib_price = s_high - lvl * rng
                if abs(curr - fib_price) <= tol:
                    fib_bull[i] = 1
                    break
        else:
            # Swing DOWN: high lebih dulu → low = retracement ke atas = resistance
            for lvl in FIB_LEVELS:
                fib_price = s_low + lvl * rng
                if abs(curr - fib_price) <= tol:
                    fib_bear[i] = 1
                    break

    idx = df.index
    return pd.Series(fib_bull, index=idx), pd.Series(fib_bear, index=idx)


def calc_snd_zones(df: pd.DataFrame,
                   lookback: int = 50,
                   atr_mult: float = 1.5) -> tuple:
    """
    Supply & Demand Zones — Order Block detection (Smart Money).

    Demand zone (bullish OB):
      Candle bearish/netral SEBELUM impulse bullish (≥ atr_mult × ATR dalam 3 candle)
      → ketika harga kembali ke zona ini = sinyal BUY

    Supply zone (bearish OB):
      Candle bullish/netral SEBELUM impulse bearish (≥ atr_mult × ATR dalam 3 candle)
      → ketika harga kembali ke zona ini = sinyal SELL

    Return: (snd_bull, snd_bear) sebagai pd.Series int (0/1)
    """
    high   = df['high'].values
    low    = df['low'].values
    close  = df['close'].values
    open_  = df['open'].values
    atr_s  = calc_atr(df, 14).values
    n      = len(df)

    # ── Identifikasi semua order block ────────────────────────────────────────
    demand_zones = []   # list of (candle_idx, zone_high, zone_low)
    supply_zones = []

    for j in range(1, n - 3):
        atr_j = atr_s[j] if not np.isnan(atr_s[j]) and atr_s[j] > 0 else 1.0

        move_up   = max(high[j+1], high[j+2], high[j+3]) - close[j]
        move_down = close[j] - min(low[j+1], low[j+2], low[j+3])

        # Demand: candle bearish/doji sebelum bullish impulse
        if move_up >= atr_mult * atr_j and close[j] <= open_[j]:
            zone_h = max(open_[j], close[j])
            zone_l = min(open_[j], close[j])
            demand_zones.append((j, zone_h, zone_l))

        # Supply: candle bullish/doji sebelum bearish impulse
        if move_down >= atr_mult * atr_j and close[j] >= open_[j]:
            zone_h = max(open_[j], close[j])
            zone_l = min(open_[j], close[j])
            supply_zones.append((j, zone_h, zone_l))

    # ── Cek apakah harga saat ini berada di dalam zone ────────────────────────
    snd_bull = np.zeros(n, dtype=int)
    snd_bear = np.zeros(n, dtype=int)

    for i in range(lookback, n):
        curr = close[i]

        for zone_idx, zone_h, zone_l in demand_zones:
            if zone_idx >= i or (i - zone_idx) > lookback:
                continue
            # Harga menyentuh zona demand (dengan sedikit toleransi)
            if zone_l * 0.998 <= curr <= zone_h * 1.005:
                snd_bull[i] = 1
                break

        for zone_idx, zone_h, zone_l in supply_zones:
            if zone_idx >= i or (i - zone_idx) > lookback:
                continue
            if zone_l * 0.995 <= curr <= zone_h * 1.002:
                snd_bear[i] = 1
                break

    idx = df.index
    return pd.Series(snd_bull, index=idx), pd.Series(snd_bear, index=idx)


# ══════════════════════════════════════════════════════════════════════════════
# POLA CANDLESTICK MANUAL
# ══════════════════════════════════════════════════════════════════════════════

def _body(o, c):           return abs(c - o)
def _upper_wick(o, h, c): return h - max(o, c)
def _lower_wick(o, l, c): return min(o, c) - l


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n  = len(df)
    bulls = np.zeros(n, dtype=int)
    bears = np.zeros(n, dtype=int)

    O = df['open'].values
    H = df['high'].values
    L = df['low'].values
    C = df['close'].values

    for i in range(n):
        o, h, l, c = O[i], H[i], L[i], C[i]
        body  = _body(o, c)
        upper = _upper_wick(o, h, c)
        lower = _lower_wick(o, l, c)
        total = h - l if h != l else 1e-9

        b, s = 0, 0

        # Hammer (bullish reversal)
        if body > 0 and lower >= 2 * body and upper <= 0.3 * body:
            b += 1

        # Shooting Star (bearish reversal)
        if c < o and body > 0 and upper >= 2 * body and lower <= 0.3 * body:
            s += 1

        # Inverted Hammer (bullish, konfirmasi dibutuhkan)
        if c > o and body > 0 and upper >= 2 * body and lower <= 0.3 * body:
            b += 1

        # Bullish Marubozu
        if c > o and body / total > 0.88:
            b += 1

        # Bearish Marubozu
        if c < o and body / total > 0.88:
            s += 1

        # Pola dua candle
        if i > 0:
            po, ph, pl, pc = O[i-1], H[i-1], L[i-1], C[i-1]

            # Bullish Engulfing
            if pc < po and c > o and o <= pc and c >= po:
                b += 2

            # Bearish Engulfing
            if pc > po and c < o and o >= pc and c <= po:
                s += 2

        # Pola tiga candle
        if i >= 2:
            f_o, f_c = O[i-2], C[i-2]
            m_o, m_c = O[i-1], C[i-1]

            # Morning Star (bullish)
            if (f_c < f_o and
                    _body(m_o, m_c) < 0.3 * _body(f_o, f_c) and
                    c > o and c > (f_o + f_c) / 2):
                b += 2

            # Evening Star (bearish)
            if (f_c > f_o and
                    _body(m_o, m_c) < 0.3 * _body(f_o, f_c) and
                    c < o and c < (f_o + f_c) / 2):
                s += 2

        bulls[i] = min(b, 3)
        bears[i] = min(s, 3)

    df['bullish_cdl'] = bulls
    df['bearish_cdl'] = bears
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung semua indikator pada DataFrame OHLCV.
    Kolom minimal: open, high, low, close, volume
    """
    df    = df.copy()
    close = df['close']

    # ── Trend ──────────────────────────────────────────────────────────────────
    df['SMA_50']  = calc_sma(close, 50)
    df['SMA_200'] = calc_sma(close, 200)
    df['EMA_10']  = calc_ema(close, 10)
    df['EMA_21']  = calc_ema(close, 21)
    df['EMA_50']  = calc_ema(close, 50)

    bb_upper, bb_mid, bb_lower = calc_bbands(close, 20)
    df['BBU_20'] = bb_upper
    df['BBM_20'] = bb_mid
    df['BBL_20'] = bb_lower

    df['MACD'], df['MACD_signal'], df['MACDh_12_26_9'] = calc_macd(close)

    psar_vals, psar_bull = calc_psar(df)
    df['PSAR']     = psar_vals
    df['sar_bull'] = psar_bull.astype(int)

    # ── Momentum ───────────────────────────────────────────────────────────────
    df['RSI_14']        = calc_rsi(close, 14)
    stoch_k, stoch_d    = calc_stochastic(df, 14, 3)
    df['STOCHk_14_3_3'] = stoch_k
    df['STOCHd_14_3_3'] = stoch_d
    df['CCI_20_0.015']  = calc_cci(df, 20)
    df['WILLR_14']      = calc_willr(df, 14)

    # ── Volatilitas ────────────────────────────────────────────────────────────
    df['ATRr_14']  = calc_atr(df, 14)
    df['STDEV_14'] = calc_stdev(close, 14)

    # ── Volume (volume=1.0 dari Deriv, hanya OBV — tidak dipakai di ML) ───────
    df['OBV'] = calc_obv(df)

    # ── Smart Money: FVG, Fibonacci, S&D ──────────────────────────────────────
    df['fvg_bull'],  df['fvg_bear']  = calc_fvg(df, lookback=30)
    df['fib_bull'],  df['fib_bear']  = calc_fibonacci(df, lookback=89, tolerance=0.002)
    df['snd_bull'],  df['snd_bear']  = calc_snd_zones(df, lookback=50, atr_mult=1.5)

    # ── Candlestick Patterns ───────────────────────────────────────────────────
    df = detect_candlestick_patterns(df)

    # ── Kolom bantu ────────────────────────────────────────────────────────────
    spread      = bb_upper - bb_lower
    df['bb_pos'] = np.where(spread > 0, (close - bb_lower) / spread, 0.5)

    df['ema_cross'] = (df['EMA_10'] > df['EMA_21']).astype(int)

    # Arah tren makro: 1 = harga di atas SMA200 (bullish), 0 = bearish
    df['trend_bull'] = (close > df['SMA_200']).astype(int)

    # Forward fill NaN lalu backward fill sisa
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    logger.debug(f"Indikator dihitung ({len(df)} baris).")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FITUR ML
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_names() -> list:
    """
    Fitur untuk Random Forest.
    MFI dihapus (volume=1.0 tidak riil → sinyal palsu).
    trend_bull ditambahkan (konteks tren makro SMA200).
    """
    return [
        'RSI_14',
        'MACDh_12_26_9',
        'ATRr_14',
        'bb_pos',
        'ema_cross',
        'STOCHk_14_3_3',
        'STOCHd_14_3_3',
        'CCI_20_0.015',
        'WILLR_14',
        'bullish_cdl',
        'bearish_cdl',
        'sar_bull',
        'trend_bull',
        # Smart Money
        'fvg_bull',
        'fvg_bear',
        'fib_bull',
        'fib_bear',
        'snd_bull',
        'snd_bear',
    ]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    features  = get_feature_names()
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]

    if missing:
        logger.warning(f"Fitur tidak tersedia: {missing}")

    result = df[available].copy()
    for col in missing:
        result[col] = 0.0

    result = result[features]
    result = result.ffill().bfill().fillna(0)
    return result
