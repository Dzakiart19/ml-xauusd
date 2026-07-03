"""
Klien WebSocket Deriv API.
- Ambil data historis OHLC (ticks_history, subscribe=1)
- Stream candle baru via msg_type='ohlc'
- Subscribe tick real-time untuk tracking TP/SL
"""

import json
import logging
import threading
import time
import websocket
import pandas as pd

from config import (
    DERIV_WS_URL, DERIV_SYMBOL,
    CANDLE_COUNT, GRANULARITY,
)

logger = logging.getLogger(__name__)


class DerivClient:
    def __init__(self, shared_state: dict, state_lock: threading.Lock,
                 on_ready_callback=None):
        self.shared_state      = shared_state
        self.state_lock        = state_lock
        self.on_ready_callback = on_ready_callback

        self.ws        = None
        self._thread   = None
        self._stop_evt = threading.Event()

        # Guard: on_ready hanya dipanggil sekali
        self._ready_called = False
        self._ready_lock   = threading.Lock()

        # Request IDs
        self._hist_req_id = 1   # ticks_history (subscribe=1 → dapat ohlc stream)
        self._tick_req_id = 2   # ticks (bid/ask real-time)

        # Retry delay — reset setelah berhasil konek
        self._retry_delay = 2

    # ─── Public ───────────────────────────────────────────────────────────────

    def start(self):
        self._thread = threading.Thread(
            target=self._run_forever, name="DerivWS", daemon=True
        )
        self._thread.start()
        logger.info("DerivClient thread dimulai.")

    def stop(self):
        self._stop_evt.set()
        if self.ws:
            self.ws.close()

    # ─── Loop koneksi dengan reconnect ────────────────────────────────────────

    def _run_forever(self):
        while not self._stop_evt.is_set():
            try:
                logger.info("Menghubungkan ke Deriv WebSocket...")
                self.ws = websocket.WebSocketApp(
                    DERIV_WS_URL,
                    on_open    = self._on_open,
                    on_message = self._on_message,
                    on_error   = self._on_error,
                    on_close   = self._on_close,
                )
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.error(f"WebSocket error tidak terduga: {e}")

            if not self._stop_evt.is_set():
                logger.info(f"Reconnect dalam {self._retry_delay}s...")
                time.sleep(self._retry_delay)
                self._retry_delay = min(self._retry_delay * 2, 60)

    # ─── WebSocket callbacks ──────────────────────────────────────────────────

    def _on_open(self, ws):
        self._retry_delay = 2   # Reset backoff setelah berhasil konek
        logger.info("WebSocket terhubung. Meminta data historis + subscribe OHLC...")
        self._request_history_and_subscribe()

    def _request_history_and_subscribe(self):
        """
        Satu request: historis + subscribe ke stream OHLC candle baru.
        Setelah konek ulang, candles diperbarui tanpa memicu on_ready ulang.
        """
        payload = {
            "ticks_history": DERIV_SYMBOL,
            "adjust_start_time": 1,
            "count": CANDLE_COUNT,
            "end": "latest",
            "granularity": GRANULARITY,
            "style": "candles",
            "subscribe": 1,
            "req_id": self._hist_req_id,
        }
        self.ws.send(json.dumps(payload))

    def _subscribe_ticks(self):
        """Subscribe tick real-time untuk tracking harga bid/ask."""
        payload = {
            "ticks": DERIV_SYMBOL,
            "subscribe": 1,
            "req_id": self._tick_req_id,
        }
        self.ws.send(json.dumps(payload))
        logger.info("Berlangganan tick real-time.")

    def _on_message(self, ws, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return

        msg_type = msg.get("msg_type")

        # ── Candle historis (jawaban awal ticks_history) ───────────────────
        if msg_type == "candles":
            candles = msg.get("candles", [])
            if not candles:
                logger.warning("Response candles kosong.")
                return

            df = self._parse_candles(candles)
            with self.state_lock:
                self.shared_state["candles"]       = df
                self.shared_state["candles_dirty"] = True

            logger.info(f"Data historis diterima: {len(df)} candle.")

            # Panggil on_ready HANYA sekali
            with self._ready_lock:
                if not self._ready_called:
                    self._ready_called = True
                    fire = True
                else:
                    fire = False

            if fire and self.on_ready_callback:
                threading.Thread(
                    target=self.on_ready_callback,
                    daemon=True,
                    name="OnReadyCB",
                ).start()

            self._subscribe_ticks()

        # ── OHLC update (candle baru / candle sedang terbentuk) ───────────
        elif msg_type == "ohlc":
            ohlc = msg.get("ohlc", {})
            self._handle_ohlc_update(ohlc)

        # ── Tick real-time (bid/ask) ───────────────────────────────────────
        elif msg_type == "tick":
            tick = msg.get("tick", {})
            bid  = tick.get("bid")
            ask  = tick.get("ask")
            if bid is not None and ask is not None:
                mid = (float(bid) + float(ask)) / 2
                with self.state_lock:
                    self.shared_state["current_bid"]   = float(bid)
                    self.shared_state["current_ask"]   = float(ask)
                    self.shared_state["current_price"] = mid

        # ── Error dari Deriv API ───────────────────────────────────────────
        elif "error" in msg:
            logger.error(f"Deriv API error: {msg['error']}")

    def _handle_ohlc_update(self, ohlc: dict):
        """
        Update atau tambahkan candle di shared_state berdasarkan OHLC update.
        candles_dirty HANYA di-set True saat candle BARU muncul (bukan update
        candle yang sedang terbentuk), agar kalkulasi indikator tidak
        dipanggil terlalu sering dan tidak menyebabkan APScheduler warning.
        """
        try:
            open_time = ohlc.get("open_time") or ohlc.get("epoch")
            if not open_time:
                return

            candle_idx = pd.Timestamp(int(open_time), unit="s", tz="UTC")
            new_row = {
                "open":   float(ohlc["open"]),
                "high":   float(ohlc["high"]),
                "low":    float(ohlc["low"]),
                "close":  float(ohlc["close"]),
                "volume": 1.0,
            }

            with self.state_lock:
                df = self.shared_state.get("candles")
                if df is None:
                    return

                if candle_idx in df.index:
                    # Update candle yang sedang terbentuk — JANGAN set dirty
                    # (indikator tidak perlu dihitung ulang untuk candle belum close)
                    for col, val in new_row.items():
                        df.at[candle_idx, col] = val
                    self.shared_state["candles"] = df
                    # candles_dirty TIDAK diubah di sini

                else:
                    # Candle 5-menit baru dimulai → set dirty untuk recalculate
                    new_df = pd.DataFrame([new_row], index=[candle_idx])
                    df = pd.concat([df, new_df])
                    if len(df) > CANDLE_COUNT:
                        df = df.iloc[-CANDLE_COUNT:]
                    self.shared_state["candles"]       = df
                    self.shared_state["candles_dirty"] = True   # ← hanya di sini
                    logger.info(
                        f"Candle baru [{candle_idx}] close={new_row['close']:.2f}"
                    )

        except (KeyError, ValueError) as e:
            logger.warning(f"OHLC update tidak valid: {e} — data: {ohlc}")

    @staticmethod
    def _parse_candles(candles: list) -> pd.DataFrame:
        df = pd.DataFrame(candles)
        df.rename(columns={"epoch": "time"}, inplace=True)
        df["time"]   = pd.to_datetime(df["time"], unit="s", utc=True)
        df["volume"] = 1.0
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, code, reason):
        logger.warning(f"WebSocket ditutup (code={code}, reason={reason}).")
