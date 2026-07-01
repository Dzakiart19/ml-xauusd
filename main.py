"""
Entry point utama — Bot Sinyal XAUUSD
Mengorkestrasi: Deriv WebSocket, Signal Generator, dan Telegram Bot.
"""

# Patch websocket-client sebelum import lain
from websocket_patch import ensure_websocket_client
ensure_websocket_client()

import logging
import threading

from config import LOG_FORMAT, LOG_LEVEL
from database import init_db
from deriv_client import DerivClient
from signal_generator import SignalGenerator
from bot import enqueue_message, run_bot

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
)
logger = logging.getLogger(__name__)

# Kurangi noise dari library pihak ketiga
logging.getLogger("websocket").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)


def main():
    logger.info("═" * 50)
    logger.info("  Bot XAUUSD Signal dimulai")
    logger.info("═" * 50)

    # 1. Inisialisasi database
    init_db()

    # 2. Shared state antar thread
    shared_state = {
        "candles":                 None,   # pd.DataFrame raw
        "candles_with_indicators": None,   # pd.DataFrame + indikator
        "current_price":           None,
        "current_bid":             None,
        "current_ask":             None,
        "is_ready":                False,
    }
    state_lock = threading.Lock()

    # 3. Signal generator (dibuat dulu, belum start)
    signal_gen = SignalGenerator(
        shared_state=shared_state,
        state_lock=state_lock,
        send_message_fn=enqueue_message,
    )

    # 4. Callback saat data historis siap
    def on_data_ready():
        logger.info("Data historis siap. Melatih model awal...")
        signal_gen.initial_train()
        signal_gen.start()

    # 5. Deriv WebSocket client
    deriv = DerivClient(
        shared_state=shared_state,
        state_lock=state_lock,
        on_ready_callback=on_data_ready,
    )
    deriv.start()

    # 6. Jalankan Telegram bot (blocking — harus di main thread)
    try:
        run_bot()
    except KeyboardInterrupt:
        logger.info("Bot dihentikan oleh pengguna.")
    finally:
        deriv.stop()
        signal_gen.stop()
        logger.info("Bot selesai.")


if __name__ == "__main__":
    main()
