#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
#  install.sh — Setup dependencies Bot XAUUSD Signal
#  Jalankan sekali: bash install.sh
# ═══════════════════════════════════════════════════════

set -e

echo "════════════════════════════════════════"
echo "  Setup Bot XAUUSD Signal"
echo "════════════════════════════════════════"

# ── Cek Python 3.8+ ──────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "❌ Python3 tidak ditemukan. Install Python 3.8+ terlebih dahulu."
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PY_VER ditemukan"

# ── Deteksi environment (Nix/Replit vs normal) ────────
if python3 -m pip install --help 2>&1 | grep -q "break-system-packages" && \
   python3 -m pip install --dry-run pip 2>&1 | grep -q "externally-managed"; then
    PIP_FLAGS="--break-system-packages --quiet"
else
    PIP_FLAGS="--quiet"
fi

# ── Install dependencies ──────────────────────────────
echo ""
echo "📦 Install dependencies dari requirements.txt..."
python3 -m pip install $PIP_FLAGS -r requirements.txt
echo "✅ Semua package terinstall"

# ── Verifikasi semua package ──────────────────────────
echo ""
echo "🔍 Verifikasi package..."

check_pkg() {
    local import_name=$1
    local display_name=$2
    if python3 -c "import $import_name" 2>/dev/null; then
        local ver
        ver=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('$display_name'))" 2>/dev/null || echo "ok")
        echo "   ✅ $display_name ($ver)"
        return 0
    else
        echo "   ❌ $display_name — GAGAL"
        return 1
    fi
}

ALL_OK=true
check_pkg "telegram"  "python-telegram-bot" || ALL_OK=false
check_pkg "websocket" "websocket-client"    || ALL_OK=false
check_pkg "pandas"    "pandas"              || ALL_OK=false
check_pkg "sklearn"   "scikit-learn"        || ALL_OK=false
check_pkg "numpy"     "numpy"              || ALL_OK=false
check_pkg "joblib"    "joblib"             || ALL_OK=false

# ── Cek environment variable ──────────────────────────
echo ""
echo "🔑 Cek environment variable..."
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "   ⚠️  TELEGRAM_BOT_TOKEN belum diset"
    echo "      → Di Replit: buka Secrets → tambah TELEGRAM_BOT_TOKEN"
    echo "      → Di server lain: export TELEGRAM_BOT_TOKEN=your_token"
else
    echo "   ✅ TELEGRAM_BOT_TOKEN ditemukan"
fi

# ── Inisialisasi database ─────────────────────────────
echo ""
echo "🗄️  Inisialisasi database..."
python3 -c "from database import init_db; init_db()" && echo "   ✅ trades.db siap"

# ── Selesai ───────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
if $ALL_OK; then
    echo "  ✅ Setup selesai! Jalankan bot dengan:"
    echo "     python3 main.py"
else
    echo "  ⚠️  Ada package yang gagal diinstall."
    echo "  Coba jalankan manual:"
    echo "     pip install -r requirements.txt --break-system-packages"
fi
echo "════════════════════════════════════════"
