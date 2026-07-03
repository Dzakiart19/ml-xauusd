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
    conn.execute("PRAGMA journal_mode=WAL")   # aman untuk multi-thread
    return conn


def init_db():
    """Buat tabel jika belum ada, jalankan migrasi schema."""
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
                ml_proba      REAL,
                source        TEXT    DEFAULT 'live',
                trend_bull    INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                win_rate   REAL NOT NULL,
                total      INTEGER NOT NULL,
                wins       INTEGER NOT NULL,
                losses     INTEGER NOT NULL,
                source     TEXT DEFAULT 'live'
            )
        """)
        conn.commit()

        # ── Migrasi: tambah kolom baru jika belum ada (database lama) ──
        _migrate_add_column(conn, "trades",      "source",     "TEXT    DEFAULT 'live'")
        _migrate_add_column(conn, "trades",      "trend_bull", "INTEGER DEFAULT 0")
        _migrate_add_column(conn, "evaluations", "source",     "TEXT    DEFAULT 'live'")

        logger.info("Database diinisialisasi.")
    finally:
        conn.close()


def _migrate_add_column(conn, table: str, column: str, definition: str):
    """Tambahkan kolom baru ke tabel jika belum ada (idempotent)."""
    try:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            conn.commit()
            logger.info(f"Migrasi: kolom '{column}' ditambahkan ke tabel '{table}'.")
    except Exception as e:
        logger.warning(f"Migrasi kolom '{column}' di '{table}': {e}")


def log_trade(trade: dict) -> int:
    """Simpan trade baru, kembalikan ID-nya."""
    conn = get_connection()
    try:
        cur = conn.execute("""
            INSERT INTO trades (
                timestamp, direction, entry_price, tp, sl,
                rsi, macd_hist, atr, bb_pos, ema_signal,
                stoch_k, stoch_d, cci, willr, mfi,
                bullish_cdl, bearish_cdl, ensemble_score, ml_proba, source, trend_bull
            ) VALUES (
                :timestamp, :direction, :entry_price, :tp, :sl,
                :rsi, :macd_hist, :atr, :bb_pos, :ema_signal,
                :stoch_k, :stoch_d, :cci, :willr, :mfi,
                :bullish_cdl, :bearish_cdl, :ensemble_score, :ml_proba, :source, :trend_bull
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
    """Ambil statistik ringkasan trade — pisahkan live vs backtest."""
    conn = get_connection()
    try:
        # Statistik live saja
        row_live = conn.execute("""
            SELECT
                COUNT(*)                                        AS total,
                SUM(CASE WHEN outcome = 'WIN'  THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN outcome = 'LOSE' THEN 1 ELSE 0 END) AS losses
            FROM trades
            WHERE outcome IS NOT NULL AND (source = 'live' OR source IS NULL)
        """).fetchone()

        # Statistik backtest (untuk referensi)
        row_bt = conn.execute("""
            SELECT
                COUNT(*)                                        AS total,
                SUM(CASE WHEN outcome = 'WIN'  THEN 1 ELSE 0 END) AS wins
            FROM trades
            WHERE outcome IS NOT NULL AND source = 'backtest'
        """).fetchone()

        total_live  = row_live["total"]  or 0
        wins_live   = row_live["wins"]   or 0
        losses_live = row_live["losses"] or 0
        rate_live   = (wins_live / total_live * 100) if total_live > 0 else 0.0

        bt_total = row_bt["total"] or 0
        bt_wins  = row_bt["wins"]  or 0

        # Evaluasi terakhir (live)
        eval_row = conn.execute("""
            SELECT timestamp, win_rate FROM evaluations
            WHERE source = 'live' OR source IS NULL
            ORDER BY id DESC LIMIT 1
        """).fetchone()

        if eval_row:
            last_eval = eval_row["timestamp"]
            last_rate = eval_row["win_rate"]
        elif total_live > 0:
            last_eval = "Data live"
            last_rate = round(rate_live, 2)
        else:
            last_eval = "Belum ada"
            last_rate = 0.0

        return {
            "total":      total_live,
            "wins":       wins_live,
            "losses":     losses_live,
            "win_rate":   round(rate_live, 2),
            "last_eval":  last_eval,
            "last_rate":  round(last_rate, 2),
            "bt_total":   bt_total,
            "bt_wins":    bt_wins,
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


def count_completed_live_trades() -> int:
    """Hitung trade live yang sudah selesai (untuk trigger retrain)."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM trades "
            "WHERE outcome IS NOT NULL AND (source = 'live' OR source IS NULL)"
        ).fetchone()
        return row["n"]
    finally:
        conn.close()


def load_all_trades_for_training():
    """
    Kembalikan list dict semua trade yang sudah selesai,
    siap dipakai untuk melatih ulang model.
    Prioritaskan trade live; sertakan backtest jika live < 30.
    """
    conn = get_connection()
    try:
        live_rows = conn.execute("""
            SELECT * FROM trades
            WHERE outcome IS NOT NULL AND (source = 'live' OR source IS NULL)
        """).fetchall()

        live_trades = [dict(r) for r in live_rows]

        if len(live_trades) >= 30:
            return live_trades

        # Kurang data live — sertakan backtest sebagai warmup
        all_rows = conn.execute("""
            SELECT * FROM trades WHERE outcome IS NOT NULL
        """).fetchall()
        return [dict(r) for r in all_rows]
    finally:
        conn.close()


def get_trade_history(limit: int = 10) -> list:
    """Ambil N trade live terakhir yang sudah selesai."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT timestamp, direction, entry_price, tp, sl,
                   outcome, pips, ensemble_score, ml_proba, source
            FROM trades
            WHERE outcome IS NOT NULL AND (source = 'live' OR source IS NULL)
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def log_evaluation(win_rate: float, total: int, wins: int, losses: int,
                   source: str = "live"):
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO evaluations (timestamp, win_rate, total, wins, losses, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now(WIB).isoformat(), win_rate, total, wins, losses, source))
        conn.commit()
    finally:
        conn.close()
