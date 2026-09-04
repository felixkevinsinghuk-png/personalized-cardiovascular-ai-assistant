#!/usr/bin/env bash
SSD_ROOT="$(cd "$(dirname "$0")" && pwd)"
MYSQL_SOCKET="$SSD_ROOT/mysql_data/mysql.sock"
LOG_DIR="$SSD_ROOT/logs"
OLLAMA_MODELS_DIR="$SSD_ROOT/ollama_models"

mkdir -p "$LOG_DIR"
mkdir -p "$OLLAMA_MODELS_DIR"

echo "Starting services for RetinalXAI..."

# --- MySQL ---
MYSQLD_PATH=$(which mysqld 2>/dev/null || brew --prefix mysql 2>/dev/null | xargs -I{} echo {}/bin/mysqld || echo "")
if [ ! -z "$MYSQLD_PATH" ] && [ ! -S "$MYSQL_SOCKET" ]; then
    echo "Starting MySQL..."
    "$MYSQLD_PATH" --defaults-file="$SSD_ROOT/mysql_ssd.cnf" --user="$(whoami)" --daemonize 2>>"$LOG_DIR/mysql_error.log"
fi

# --- Ollama (system install, but models live on SSD) ---
export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
if ! pgrep -x "ollama" >/dev/null; then
    echo "Starting Ollama (models on SSD: $OLLAMA_MODELS_DIR)..."
    nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    sleep 2
fi

echo "Services started."
