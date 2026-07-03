"""
Patch startup untuk memastikan websocket-client (bukan stub) yang digunakan.
Diimpor paling awal di main.py sebelum modul lain.

Strategi: cek dulu apakah WebSocketApp sudah tersedia.
Patch __init__.py hanya jika memang bermasalah, dan hanya jika file _app.py ada
(artinya websocket-client memang terinstal, bukan library lain).
"""

import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)

_INIT_CONTENT = '''\
"""websocket-client __init__.py (auto-restored by websocket_patch)"""
from ._core import WebSocket, create_connection
from ._app import WebSocketApp
from ._exceptions import (
    WebSocketException,
    WebSocketProtocolException,
    WebSocketPayloadException,
    WebSocketConnectionClosedException,
    WebSocketTimeoutException,
    WebSocketProxyException,
    WebSocketBadStatusException,
    WebSocketAddressException,
)
from ._abnf import ABNF
from ._logging import enableTrace
__version__ = "1.9.0"
'''


def _find_websocket_client_path() -> str | None:
    """Cari direktori websocket-client di sys.path (harus punya _app.py)."""
    for p in sys.path:
        candidate = os.path.join(p, 'websocket')
        if (os.path.isdir(candidate) and
                os.path.exists(os.path.join(candidate, '_app.py'))):
            return candidate
    return None


def ensure_websocket_client():
    """
    Pastikan modul 'websocket' mengekspos WebSocketApp dari websocket-client.
    Jika sudah benar, tidak melakukan apa-apa.
    Jika rusak, perbaiki __init__.py hanya jika websocket-client terinstal.
    """
    try:
        # Bersihkan cache modul agar import ulang dari disk
        if 'websocket' in sys.modules:
            del sys.modules['websocket']
        importlib.invalidate_caches()

        import websocket as _ws
        if hasattr(_ws, 'WebSocketApp'):
            logger.debug("websocket-client sudah aktif — tidak perlu patch.")
            return

        # WebSocketApp tidak ditemukan — perlu patch
        ws_path = _find_websocket_client_path()
        if not ws_path:
            logger.error(
                "websocket-client tidak ditemukan di sys.path. "
                "Install dengan: pip install websocket-client"
            )
            return

        init_file = os.path.join(ws_path, '__init__.py')

        # Baca isi saat ini — patch hanya jika isi berbeda
        current = ""
        try:
            with open(init_file) as f:
                current = f.read()
        except Exception:
            pass

        if current.strip() == _INIT_CONTENT.strip():
            # Isi sudah benar, reload saja
            pass
        else:
            logger.info(f"Memperbaiki {init_file} ...")
            with open(init_file, 'w') as f:
                f.write(_INIT_CONTENT)

        # Reload
        if 'websocket' in sys.modules:
            del sys.modules['websocket']
        importlib.invalidate_caches()
        import websocket as _ws2  # noqa: F811

        if hasattr(_ws2, 'WebSocketApp'):
            logger.info("websocket-client berhasil di-patch.")
        else:
            logger.error(
                "Patch websocket-client gagal. "
                "Coba reinstall: pip install --force-reinstall websocket-client"
            )

    except Exception as e:
        logger.error(f"websocket_patch error: {e}")
