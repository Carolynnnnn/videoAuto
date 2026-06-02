#!/bin/bash
# AI Video Pipeline - GUI Launcher
# Usage: ./launch_gui.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "  AI Video Pipeline - GUI Launcher"
echo "======================================"

# Select Python interpreter
if command -v python3.11 &> /dev/null; then
    PYTHON_BIN="python3.11"
    echo "Using Python 3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
    echo "Using Python 3 (fallback from python3.11)"
else
    echo "Error: Neither python3.11 nor python3 found, please install Python 3"
    exit 1
fi

# Check PyQt5
"$PYTHON_BIN" -c "from PyQt5.QtWidgets import QApplication" 2>/dev/null || {
    echo "Installing PyQt5..."
    sudo pip3 install PyQt5
}

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg not found, please install FFmpeg"
    exit 1
fi

# Setup API Keys (can override via environment variables)
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export MINIMAX_API_KEY="${MINIMAX_API_KEY:-}"

echo "DeepSeek API: ${DEEPSEEK_API_KEY:+[configured]}"
echo "Minimax API: ${MINIMAX_API_KEY:+[configured]}"
echo ""
echo "Launching GUI..."

# Launch new GUI entrypoint
"$PYTHON_BIN" -m src.gui.app
