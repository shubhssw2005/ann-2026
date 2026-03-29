#!/usr/bin/env bash
# Full pipeline: build → fetch → features → train → start live streams
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Detect actual cargo target dir
CARGO_TARGET="$(cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['target_directory'])" 2>/dev/null || echo "$ROOT/target")"
BIN_DIR="$CARGO_TARGET/release"
echo "[info] binaries at: $BIN_DIR"

echo "=== Multi-Coin ANN Pipeline (Top ${TOP_N:-100} coins) ==="

# 1. Build both Rust binaries
echo "[1/4] Building Rust binaries..."
cargo build --release

# 2. Fetch historical data for all top coins
echo "[2/4] Fetching historical market data for top ${TOP_N:-100} coins..."
INTERVAL=${INTERVAL:-15m} LIMIT=${LIMIT:-1500} TOP_N=${TOP_N:-100} "$BIN_DIR/btc-fetcher"

# 3. Feature engineering + EDA
echo "[3/4] Engineering features across all coins..."
python pipeline/features.py

# 4. Train model
echo "[4/4] Training ANN on combined dataset..."
python pipeline/train.py

echo ""
echo "=== Pipeline complete ==="
echo ""
echo "To go live, open TWO terminals:"
echo ""
echo "  Terminal 1 — WebSocket stream (real-time candle data):"
echo "    cd $(pwd) && INTERVAL=${INTERVAL:-15m} $BIN_DIR/ws-stream"
echo ""
echo "  Terminal 2 — Scheduler (fires predict on every candle close):"
echo "    cd $(pwd) && python scripts/scheduler.py ws"
echo ""
echo "  Or REST polling fallback (no WebSocket):"
echo "    cd $(pwd) && python scripts/scheduler.py poll"
echo ""
echo "  Single coin predict:"
echo "    python pipeline/predict.py ETHUSDT"
