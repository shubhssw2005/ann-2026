"""
FastAPI server — serves live signals + backtest results to frontend
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import json, subprocess, sys

app = FastAPI()

BASE = Path(__file__).parent.parent
STATIC = Path(__file__).parent / "src"

# ─── API routes ───────────────────────────────────────────────────────────────


@app.get("/api/signals")
def get_signals():
    p = BASE / "data/processed/latest_signals.json"
    if not p.exists():
        return JSONResponse({"error": "No signals yet. Run predict.py first."}, 404)
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/backtest")
def get_backtest():
    p = BASE / "data/backtest/results.json"
    if not p.exists():
        return JSONResponse({"error": "No backtest yet. Run backtest.py first."}, 404)
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/coin/{symbol}")
def get_coin(symbol: str):
    p = BASE / "data/backtest/results.json"
    if not p.exists():
        return JSONResponse({"error": "No backtest data"}, 404)
    data = json.loads(p.read_text())
    coin = next((c for c in data["coins"] if c["symbol"] == symbol.upper()), None)
    if not coin:
        return JSONResponse({"error": f"{symbol} not found"}, 404)
    return JSONResponse(coin)


@app.get("/api/status")
def status():
    model_exists = (BASE / "models/best_model.pt").exists()
    signals_exists = (BASE / "data/processed/latest_signals.json").exists()
    backtest_exists = (BASE / "data/backtest/results.json").exists()
    return {
        "model_trained": model_exists,
        "signals_ready": signals_exists,
        "backtest_ready": backtest_exists,
    }


# ─── Static frontend ──────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
