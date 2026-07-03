"""
Telegram bot handler.
Perintah: /start, /stop, /ping, /stats, /history
Pesan sinyal dikirim via internal queue (thread-safe).
"""

import json
import logging
import os
import queue
import tempfile
import threading
from datetime import datetime, timezone, timedelta

WIB = timezone(timedelta(hours=7))

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import Conflict, InvalidToken, NetworkError, RetryAfter
from telegram.ext import (
    Application, ApplicationBuilder,
    CommandHandler, ContextTypes,
)

from config import TELEGRAM_TOKEN, CHAT_IDS_FILE
from database import get_stats, get_trade_history

logger = logging.getLogger(__name__)

# ─── Waktu mulai bot ──────────────────────────────────────────────────────────
_BOT_START_TIME = datetime.now(WIB)


def _format_uptime() -> str:
    delta         = datetime.now(WIB) - _BOT_START_TIME
    total_seconds = int(delta.total_seconds())
    days    = total_seconds // 86400
    hours   = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    parts = []
    if days:    parts.append(f"{days}h")
    if hours:   parts.append(f"{hours}j")
    if minutes: parts.append(f"{minutes}m")
    parts.append(f"{seconds}d")
    return " ".join(parts)


# ─── Queue pesan keluar (thread-safe) ─────────────────────────────────────────
_msg_queue: queue.Queue = queue.Queue()

# ─── Sinyal aktif (diupdate oleh SignalGenerator) ─────────────────────────────
_active_signal: dict | None = None
_active_signal_lock = threading.Lock()


def set_active_signal(trade: dict | None):
    global _active_signal
    with _active_signal_lock:
        _active_signal = trade


def get_active_signal() -> dict | None:
    with _active_signal_lock:
        return _active_signal


def enqueue_message(text: str):
    """Dipanggil dari thread mana saja untuk broadcast ke semua chat terdaftar."""
    _msg_queue.put(text)


# ─── Manajemen chat_id ────────────────────────────────────────────────────────

def _load_chat_ids() -> set:
    if not os.path.exists(CHAT_IDS_FILE):
        return set()
    try:
        with open(CHAT_IDS_FILE) as f:
            return set(json.load(f))
    except Exception:
        return set()


def _save_chat_ids(ids: set):
    """Atomic write: tulis ke file sementara lalu rename, agar tidak corrupt."""
    try:
        dir_name = os.path.dirname(os.path.abspath(CHAT_IDS_FILE))
        with tempfile.NamedTemporaryFile(
            mode="w", dir=dir_name, delete=False, suffix=".tmp"
        ) as tf:
            json.dump(list(ids), tf)
            tmp_path = tf.name
        os.replace(tmp_path, CHAT_IDS_FILE)
    except Exception as e:
        logger.error(f"Gagal menyimpan chat_ids: {e}")


_chat_ids      : set            = _load_chat_ids()
_chat_ids_lock : threading.Lock = threading.Lock()


def get_chat_ids() -> list:
    with _chat_ids_lock:
        return list(_chat_ids)


def register_chat(chat_id: int):
    with _chat_ids_lock:
        _chat_ids.add(chat_id)
        _save_chat_ids(_chat_ids)


def unregister_chat(chat_id: int):
    with _chat_ids_lock:
        _chat_ids.discard(chat_id)
        _save_chat_ids(_chat_ids)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _escape_md(text: str) -> str:
    """Escape semua karakter khusus MarkdownV2."""
    for ch in r"_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


# ─── Command handlers ─────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    register_chat(chat_id)

    active = get_active_signal()
    if active:
        direction = active["direction"]
        arrow     = "🟢" if direction == "BUY" else "🔴"
        entry     = _escape_md(f"{active['entry_price']:.2f}")
        tp        = _escape_md(f"{active['tp']:.2f}")
        sl        = _escape_md(f"{active['sl']:.2f}")
        since     = _escape_md(str(active.get("timestamp", ""))[:16].replace("T", " "))
        signal_block = (
            f"\n\n{arrow} *Sinyal Aktif Sekarang:*\n"
            f"Arah: `{direction}` \\| Sejak: `{since} WIB`\n"
            f"Entry `{entry}` ➜ TP `{tp}` SL `{sl}`"
        )
    else:
        signal_block = "\n\n⏳ Tidak ada sinyal aktif saat ini\\."

    await update.message.reply_text(
        "✅ *Bot XAUUSD Signal aktif\\!*\n\n"
        "Kamu akan menerima sinyal trading XAUUSD secara otomatis\\."
        f"{signal_block}\n\n"
        "Perintah yang tersedia:\n"
        "• /stats   – Statistik trading live\n"
        "• /history – Riwayat 10 trade terakhir\n"
        "• /ping    – Cek status bot\n"
        "• /stop    – Berhenti menerima sinyal",
        parse_mode=ParseMode.MARKDOWN_V2,
    )
    logger.info(f"Chat {chat_id} terdaftar.")


async def cmd_stop(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    unregister_chat(chat_id)
    await update.message.reply_text(
        "🔕 Kamu tidak akan menerima sinyal lagi\\.\n"
        "Ketik /start untuk berlangganan ulang\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    start_str = _BOT_START_TIME.strftime("%Y\\-%m\\-%d %H:%M WIB")
    uptime    = _format_uptime()
    await update.message.reply_text(
        f"🟢 *Bot berjalan normal\\!* Pong 🏓\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 Aktif sejak : `{start_str}`\n"
        f"⏱ Uptime      : `{uptime}`",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    trades = get_trade_history(limit=10)
    if not trades:
        await update.message.reply_text(
            "📭 Belum ada riwayat trade live\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    lines = ["📋 *Riwayat 10 Trade Live Terakhir*\n━━━━━━━━━━━━━━━━━━━━━"]
    for t in trades:
        emoji     = "✅" if t["outcome"] == "WIN" else "❌"
        arrow     = "🟢" if t["direction"] == "BUY" else "🔴"
        pips_sign = "\\+" if t["outcome"] == "WIN" else "\\-"
        pips      = abs(t["pips"] or 0)

        ts_raw = str(t["timestamp"])[:16].replace("T", " ")
        ts     = _escape_md(ts_raw)

        entry = _escape_md(f"{t['entry_price']:.2f}")
        tp    = _escape_md(f"{t['tp']:.2f}")
        sl    = _escape_md(f"{t['sl']:.2f}")
        pip_s = _escape_md(f"{pips:.0f}")

        lines.append(
            f"{emoji} {arrow} *{t['direction']}* \\| `{ts} WIB`\n"
            f"   Entry `{entry}` ➜ TP `{tp}` SL `{sl}`\n"
            f"   Pips: `{pips_sign}{pip_s}`"
        )

    wins   = sum(1 for t in trades if t["outcome"] == "WIN")
    losses = len(trades) - wins
    lines.append(
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ WIN `{wins}` \\| ❌ LOSE `{losses}` \\| Total `{len(trades)}`"
    )
    await update.message.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    stats   = get_stats()
    bar_len = 10
    wins_bar = int((stats["win_rate"] / 100) * bar_len) if stats["total"] > 0 else 0
    bar      = "🟩" * wins_bar + "🟥" * (bar_len - wins_bar)

    last_eval = _escape_md(stats["last_eval"])

    # Baris backtest (hanya tampilkan jika ada)
    bt_line = ""
    if stats.get("bt_total", 0) > 0:
        bt_wr = stats["bt_wins"] / stats["bt_total"] * 100
        bt_line = (
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📁 *Backtest \\(referensi\\)*\n"
            f"   Total `{stats['bt_total']}` \\| WIN `{stats['bt_wins']}` \\| WR `{bt_wr:.1f}%`\n"
        )

    if stats["total"] == 0:
        body = (
            f"📊 *Statistik Trading XAUUSD*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏳ Belum ada trade live yang selesai\\.\n"
            f"Bot sedang memantau pasar\\.\n"
            f"{bt_line}"
        )
    else:
        body = (
            f"📊 *Statistik Trading XAUUSD \\(Live\\)*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Menang    : `{stats['wins']}`\n"
            f"❌ Kalah     : `{stats['losses']}`\n"
            f"📈 Total     : `{stats['total']}`\n"
            f"🏆 Win Rate  : `{stats['win_rate']:.1f}%`\n"
            f"{bar}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔄 Evaluasi: `{last_eval}`\n"
            f"{bt_line}"
        )

    await update.message.reply_text(body, parse_mode=ParseMode.MARKDOWN_V2)


# ─── Job: proses queue pesan keluar ──────────────────────────────────────────

async def _process_queue(ctx: ContextTypes.DEFAULT_TYPE):
    """APScheduler job — dijalankan setiap 1 detik."""
    while not _msg_queue.empty():
        try:
            text = _msg_queue.get_nowait()
        except queue.Empty:
            break

        chat_ids = get_chat_ids()
        if not chat_ids:
            logger.debug("Tidak ada chat terdaftar; pesan dibuang.")
            continue

        for cid in chat_ids:
            try:
                await ctx.bot.send_message(
                    chat_id=cid,
                    text=text,
                    parse_mode=ParseMode.MARKDOWN,
                )
            except RetryAfter as e:
                logger.warning(f"Rate limit — tunggu {e.retry_after}s")
            except Exception as e:
                logger.error(f"Gagal kirim ke {cid}: {e}")


# ─── Error handler global ─────────────────────────────────────────────────────

async def error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    err = ctx.error
    if isinstance(err, Conflict):
        logger.warning("Conflict: instance lain masih polling, menunggu...")
        return
    if isinstance(err, (NetworkError, RetryAfter)):
        logger.warning(f"Network/Rate error (akan retry): {err}")
        return
    logger.error(f"Unhandled error: {err}", exc_info=err)


# ─── Build & run ──────────────────────────────────────────────────────────────

def build_application() -> Application:
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN tidak ditemukan di environment!")

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .build()
    )

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("stop",    cmd_stop))
    app.add_handler(CommandHandler("ping",    cmd_ping))
    app.add_handler(CommandHandler("stats",   cmd_stats))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_error_handler(error_handler)

    # Proses queue pesan keluar setiap 1 detik
    app.job_queue.run_repeating(_process_queue, interval=1, first=3)

    return app


def run_bot():
    """
    Jalankan Telegram bot (blocking). Dipanggil dari main thread.
    Retry otomatis jika terjadi Conflict atau network error.
    """
    import time as _time

    retry_delay = 15

    while True:
        try:
            app = build_application()
            logger.info("Memulai Telegram bot polling...")
            app.run_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES,
            )
            break

        except (ValueError, InvalidToken) as e:
            logger.critical(f"Token bot tidak valid: {e}. Bot dihentikan.")
            raise

        except Conflict:
            logger.warning(
                f"Conflict: sesi polling lama belum berakhir. "
                f"Retry dalam {retry_delay}s..."
            )
            _time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 120)

        except NetworkError as e:
            logger.warning(f"NetworkError: {e}. Retry dalam 10s...")
            _time.sleep(10)

        except Exception as e:
            logger.error(f"Bot error tidak terduga: {e}. Retry dalam 15s...")
            _time.sleep(15)
