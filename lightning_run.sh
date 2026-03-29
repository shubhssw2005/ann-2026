#!/usr/bin/env bash
# Lightning AI — full pipeline from scratch (no file upload needed)
# Run this in Lightning Studio terminal after cloning the repo
set -e

echo "=== BTC-ANN on Lightning AI H200 ==="

# 1. Install deps
echo "[1/4] Installing dependencies..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q ta scikit-learn matplotlib seaborn joblib pandas numpy requests fastapi uvicorn

# 2. Verify GPU
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')
"

# 3. Fetch data from Binance directly in Python
echo "[2/4] Fetching market data from Binance..."
python pipeline/fetch_data.py

# 4. Feature engineering
echo "[3/4] Engineering features..."
python pipeline/features.py

# 5. Train on GPU
echo "[4/4] Training ANN on H200..."
python pipeline/train.py

echo ""
echo "=== Training complete ==="
echo "Run backtest:  python backtest/backtest.py"
echo "Download model: models/best_model.pt"
