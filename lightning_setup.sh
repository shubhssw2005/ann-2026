#!/usr/bin/env bash
# Lightning AI Studio setup — run this once after opening a new studio
set -e

echo "=== Setting up BTC-ANN on Lightning AI ==="

# 1. Install deps
pip install -q ta scikit-learn matplotlib seaborn joblib pandas numpy

# 2. Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# 3. Run features (data already uploaded)
echo "[1/2] Engineering features..."
python pipeline/features.py

# 4. Train
echo "[2/2] Training on GPU..."
python pipeline/train.py

echo "=== Done. Run backtest next: ==="
echo "  python backtest/backtest.py"
