"""
Manajemen database SQLite untuk pencatatan trade.
"""

import sqlite3
import logging
from datetime import datetime, timezone, timedelta

WIB = timezone(timedelta(hours=7))
from config import DB_FILE

logger = logging.getLogger(__name__)


def get_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Buat tabel jika belum ada."""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                direction     TEXT    NOT NULL,
                entry_price   REAL    NOT NULL,
                tp            REAL    NOT NULL,
                sl            REAL    NOT NULL,
                outcome       TEXT,
                pips          REAL,
                rsi           REAL,
                macd_hist     REAL,
                atr           REAL,
                bb_pos        REAL,
                ema_signal    INTEGER,
                stoch_k       REAL,
                stoch_d       REAL,
                cci           REAL,
                willr         REAL,
                mfi           REAL,
                bullish_cdl   INTEGER,
                bearish_cdl   INTEGER,
                ensemble_score INTEGER,
                ml_proba      REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                win_rate   REAL NOT NULL,
                total      INTEGER NOT NULL,
                wins       INTEGER NOT NULL,
                losses     INTEGER NOT NULL
            )
        """)
        conn.commit()
        logger.info("Database diinisialisasi.")
    finally:
        conn.close()


def log_trade(trade: dict) -> int:
    """Simpan trade baru, kembalikan ID-nya."""
    conn = get_connection()
    try:
        cur = conn.execute("""
            INSERT INTO trades (
                timestamp, direction, entry_price, tp, sl,
                rsi, macd_hist, atr, bb_pos, ema_signal,
                stoch_k, stoch_d, cci, willr, mfi,
                bullish_cdl, bearish_cdl, ensemble_score, ml_proba
            ) VALUES (
                :timestamp, :direction, :entry_price, :tp, :sl,
                :rsi, :macd_hist, :atr, :bb_pos, :ema_signal,
                :stoch_k, :stoch_d, :cci, :willr, :mfi,
                :bullish_cdl, :bearish_cdl, :ensemble_score, :ml_proba
            )
        """, trade)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_trade_outcome(trade_id: int, outcome: str, pips: float):
    """Update hasil trade (WIN/LOSE) dan pips."""
    conn = get_connection()
    try:
        conn.execute("""
            UPDATE trades SET outcome = ?, pips = ? WHERE id = ?
        """, (outcome, pips, trade_id))
        conn.commit()
    finally:
        conn.close()


def get_stats() -> dict:
    """Ambil statistik ringkasan trade."""
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT
                COUNT(*)                                   AS total,
                SUM(CASE WHEN outcome = 'WIN'  THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN outcome = 'LOSE' THEN 1 ELSE 0 END) AS losses
            FROM trades
            WHERE outcome IS NOT NULL
        """).fetchone()

        total  = row["total"]  or 0
        wins   = row["wins"]   or 0
        losses = row["losses"] or 0
        rate   = (wins / total * 100) if total > 0 else 0.0

        # Evaluasi terakhir
        eval_row = conn.execute("""
            SELECT timestamp, win_rate FROM evaluations ORDER BY id DESC LIMIT 1
        """).fetchone()

        if eval_row:
            last_eval = eval_row["timestamp"]
            last_rate = eval_row["win_rate"]
        elif total > 0:
            # Belum ada evaluasi formal — tampilkan data dari trades
            last_eval = "Data awal (backtest)"
            last_rate = round(rate, 2)
        else:
            last_eval = "Belum ada"
            last_rate = 0.0

        return {
            "total":     total,
            "wins":      wins,
            "losses":    losses,
            "win_rate":  round(rate, 2),
            "last_eval": last_eval,
            "last_rate": round(last_rate, 2),
        }
    finally:
        conn.close()


def count_completed_trades() -> int:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE outcome IS NOT NULL"
        ).fetchone()
        return row["n"]
    finally:
        conn.close()


def load_all_trades_for_training():
    """
    Kembalikan list dict semua trade yang sudah selesai,
    siap dipakai untuk melatih ulang model.
    """
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT * FROM trades WHERE outcome IS NOT NULL
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_trade_history(limit: int = 10) -> list:
    """Ambil N trade terakhir yang sudah selesai."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT timestamp, direction, entry_price, tp, sl,
                   outcome, pips, ensemble_score, ml_proba
            FROM trades
            WHERE outcome IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def log_evaluation(win_rate: float, total: int, wins: int, losses: int):
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO evaluations (timestamp, win_rate, total, wins, losses)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now(WIB).isoformat(), win_rate, total, wins, losses))
        conn.commit()
    finally:
        conn.close()
