"""
Manajemen database SQLite untuk pencatatan trade.
Koneksi dibuka-tutup per operasi (short-lived) dengan timeout=30 detik.
Pendekatan ini memastikan tidak ada lock yang tersisa saat proses dimatikan.
WAL mode + synchronous=NORMAL untuk performa optimal.
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta

WIB = timezone(timedelta(hours=7))
from config import DB_FILE

logger = logging.getLogger(__name__)


# ─── Helper koneksi ────────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    """
    Context manager koneksi SQLite.
    Buka → yield → commit/rollback → tutup.
    timeout=30: tunggu hingga 30 detik jika DB sedang dikunci proses lain.
    """
    conn = sqlite3.connect(DB_FILE, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Inisialisasi ─────────────────────────────────────────────────────────────

def init_db():
    """Buat tabel jika belum ada, jalankan migrasi schema."""
    with _get_conn() as conn:
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
                trend_bull    INTEGER DEFAULT 0,
                sar_bull      INTEGER DEFAULT 0
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

    _run_migrations()
    logger.info("Database diinisialisasi.")


def _run_migrations():
    """Jalankan migrasi schema (idempotent)."""
    with _get_conn() as conn:
        _migrate_add_column(conn, "trades",      "source",     "TEXT    DEFAULT 'live'")
        _migrate_add_column(conn, "trades",      "trend_bull", "INTEGER DEFAULT 0")
        _migrate_add_column(conn, "trades",      "sar_bull",   "INTEGER DEFAULT 0")
        _migrate_add_column(conn, "evaluations", "source",     "TEXT    DEFAULT 'live'")

        cur = conn.execute("UPDATE trades SET source='live' WHERE source IS NULL")
        if cur.rowcount:
            logger.info(f"Normalisasi: {cur.rowcount} trade lama source=NULL → 'live'.")


def _migrate_add_column(conn, table: str, column: str, definition: str):
    """Tambahkan kolom baru ke tabel jika belum ada (idempotent)."""
    try:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            logger.info(f"Migrasi: kolom '{column}' ditambahkan ke tabel '{table}'.")
    except Exception as e:
        logger.warning(f"Migrasi kolom '{column}' di '{table}': {e}")


# ─── Trade CRUD ───────────────────────────────────────────────────────────────

def log_trade(trade: dict) -> int:
    """Simpan trade baru, kembalikan ID-nya."""
    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO trades (
                timestamp, direction, entry_price, tp, sl,
                rsi, macd_hist, atr, bb_pos, ema_signal,
                stoch_k, stoch_d, cci, willr, mfi,
                bullish_cdl, bearish_cdl, ensemble_score, ml_proba,
                source, trend_bull, sar_bull
            ) VALUES (
                :timestamp, :direction, :entry_price, :tp, :sl,
                :rsi, :macd_hist, :atr, :bb_pos, :ema_signal,
                :stoch_k, :stoch_d, :cci, :willr, :mfi,
                :bullish_cdl, :bearish_cdl, :ensemble_score, :ml_proba,
                :source, :trend_bull, :sar_bull
            )
        """, trade)
        return cur.lastrowid


def update_trade_outcome(trade_id: int, outcome: str, pips: float):
    """Update hasil trade (WIN/LOSE/TIMEOUT) dan pips."""
    with _get_conn() as conn:
        conn.execute("""
            UPDATE trades SET outcome = ?, pips = ? WHERE id = ?
        """, (outcome, pips, trade_id))


# ─── Statistik ────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """Ambil statistik ringkasan trade — pisahkan live vs backtest."""
    try:
        with _get_conn() as conn:
            row_live = conn.execute("""
                SELECT
                    COUNT(*)                                            AS total,
                    SUM(CASE WHEN outcome = 'WIN'  THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN outcome = 'LOSE' THEN 1 ELSE 0 END) AS losses
                FROM trades
                WHERE outcome IN ('WIN','LOSE') AND (source = 'live' OR source IS NULL)
            """).fetchone()

            row_bt = conn.execute("""
                SELECT
                    COUNT(*)                                            AS total,
                    SUM(CASE WHEN outcome = 'WIN'  THEN 1 ELSE 0 END) AS wins
                FROM trades
                WHERE outcome IN ('WIN','LOSE') AND source = 'backtest'
            """).fetchone()

            total_live  = row_live["total"]  or 0
            wins_live   = row_live["wins"]   or 0
            losses_live = row_live["losses"] or 0
            rate_live   = (wins_live / total_live * 100) if total_live > 0 else 0.0

            bt_total = row_bt["total"] or 0
            bt_wins  = row_bt["wins"]  or 0

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
    except Exception as e:
        logger.error(f"get_stats error: {e}")
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "last_eval": "Error", "last_rate": 0.0, "bt_total": 0, "bt_wins": 0}


def count_completed_live_trades() -> int:
    """Hitung trade live yang sudah selesai (WIN/LOSE) — untuk trigger retrain."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM trades "
                "WHERE outcome IN ('WIN','LOSE') AND (source = 'live' OR source IS NULL)"
            ).fetchone()
            return row["n"]
    except Exception as e:
        logger.error(f"count_completed_live_trades error: {e}")
        return 0


def count_backtest_trades() -> int:
    """Hitung trade backtest yang sudah selesai."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM trades "
                "WHERE outcome IN ('WIN','LOSE') AND source = 'backtest'"
            ).fetchone()
            return row["n"]
    except Exception as e:
        logger.error(f"count_backtest_trades error: {e}")
        return 0


def reset_backtest_trades():
    """
    Hapus semua trade backtest (source='backtest') dari database.
    Trade live (source='live') TIDAK tersentuh sama sekali.
    """
    try:
        with _get_conn() as conn:
            cur = conn.execute("DELETE FROM trades WHERE source = 'backtest'")
            deleted = cur.rowcount
        if deleted:
            logger.info(f"Reset backtest: {deleted} trade lama dihapus.")
        else:
            logger.info("Reset backtest: tidak ada data lama.")
    except Exception as e:
        logger.error(f"reset_backtest_trades error: {e}")
        raise


_LIVE_ONLY_THRESHOLD = 200   # live trade >= angka ini → backtest tidak dipakai

def load_all_trades_for_training() -> list:
    """
    Kembalikan list dict semua trade yang sudah selesai, siap untuk training.

    Policy:
    - Belum ada live trade         → backtest saja (pure warmup)
    - 1–199 live trade             → backtest + live×5 (live lebih berbobot)
    - >= 200 live trade            → live saja (cukup data nyata; buang backtest)
    """
    try:
        with _get_conn() as conn:
            live_rows = conn.execute("""
                SELECT * FROM trades
                WHERE outcome IN ('WIN','LOSE') AND source = 'live'
            """).fetchall()
            live_trades = [dict(r) for r in live_rows]

            if len(live_trades) >= _LIVE_ONLY_THRESHOLD:
                return live_trades

            bt_rows = conn.execute("""
                SELECT * FROM trades
                WHERE outcome IN ('WIN','LOSE') AND source = 'backtest'
            """).fetchall()
            bt_trades = [dict(r) for r in bt_rows]

        if not live_trades:
            return bt_trades

        return bt_trades + live_trades * 5
    except Exception as e:
        logger.error(f"load_all_trades_for_training error: {e}")
        return []


def get_trade_history(limit: int = 10) -> list:
    """Ambil N trade live terakhir yang sudah selesai (WIN/LOSE)."""
    try:
        with _get_conn() as conn:
            rows = conn.execute("""
                SELECT timestamp, direction, entry_price, tp, sl,
                       outcome, pips, ensemble_score, ml_proba, source
                FROM trades
                WHERE outcome IN ('WIN','LOSE') AND (source = 'live' OR source IS NULL)
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"get_trade_history error: {e}")
        return []


def log_evaluation(win_rate: float, total: int, wins: int, losses: int,
                   source: str = "live"):
    try:
        with _get_conn() as conn:
            conn.execute("""
                INSERT INTO evaluations (timestamp, win_rate, total, wins, losses, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (datetime.now(WIB).isoformat(), win_rate, total, wins, losses, source))
    except Exception as e:
        logger.error(f"log_evaluation error: {e}")
