"""
Live Inference — Multi-Coin ANN
Runs prediction on all tracked coins or a specific one.
"""

import numpy as np
import torch
import joblib, json, sys
from pathlib import Path
from model import BTCANN
from features import load_coin, build_features, FEATURE_COLS

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}
EMOJI = {0: "🔴", 1: "⚪", 2: "🟢"}


def load_model_and_meta():
    ckpt = torch.load("models/best_model.pt", map_location=DEVICE)
    model = BTCANN(n_features=ckpt["n_features"], n_classes=3).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = joblib.load("models/scaler.pkl")
    with open("data/processed/feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols


def predict_coin(symbol: str, model, scaler, feature_cols) -> dict | None:
    df = load_coin(symbol)
    if df is None or len(df) < 100:
        return None

    df = build_features(df)
    available = [c for c in feature_cols if c in df.columns]
    row = df[available].dropna()
    if len(row) == 0:
        return None

    X = scaler.transform(row.iloc[[-1]].values.astype(np.float32))
    with torch.no_grad():
        probs = (
            torch.softmax(model(torch.from_numpy(X).to(DEVICE)), dim=-1)
            .cpu()
            .numpy()[0]
        )

    signal = LABELS[int(np.argmax(probs))]
    return {
        "symbol": symbol,
        "timestamp": str(df.index[-1]),
        "close": float(df["close"].iloc[-1]),
        "signal": signal,
        "probs": {
            "SELL": float(probs[0]),
            "HOLD": float(probs[1]),
            "BUY": float(probs[2]),
        },
        "confidence": float(np.max(probs)),
    }


def predict_all():
    model, scaler, feature_cols = load_model_and_meta()

    with open("data/processed/processed_coins.json") as f:
        coins = json.load(f)

    print(f"\nRunning inference on {len(coins)} coins...\n")
    results = []

    for sym in coins:
        r = predict_coin(sym, model, scaler, feature_cols)
        if r:
            results.append(r)

    # Sort by confidence
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # Print summary table
    print(
        f"{'#':>3} {'Symbol':<14} {'Signal':<6} {'Conf':>6}  {'SELL':>6} {'HOLD':>6} {'BUY':>6}  {'Close':>12}"
    )
    print("-" * 75)
    for i, r in enumerate(results, 1):
        p = r["probs"]
        print(
            f"{i:>3} {r['symbol']:<14} "
            f"{EMOJI[['SELL','HOLD','BUY'].index(r['signal'])]} {r['signal']:<5} "
            f"{r['confidence']:>6.3f}  "
            f"{p['SELL']:>6.3f} {p['HOLD']:>6.3f} {p['BUY']:>6.3f}  "
            f"{r['close']:>12.4f}"
        )

    # Top BUY signals
    buys = [r for r in results if r["signal"] == "BUY"]
    sells = [r for r in results if r["signal"] == "SELL"]
    print(f"\n🟢 BUY signals  ({len(buys)}): {[r['symbol'] for r in buys[:10]]}")
    print(f"🔴 SELL signals ({len(sells)}): {[r['symbol'] for r in sells[:10]]}")

    # Save results
    with open("data/processed/latest_signals.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/processed/latest_signals.json")

    return results


if __name__ == "__main__":
    # Optional: predict single coin
    if len(sys.argv) > 1:
        sym = sys.argv[1].upper()
        model, scaler, feature_cols = load_model_and_meta()
        r = predict_coin(sym, model, scaler, feature_cols)
        if r:
            print(
                f"\n{r['symbol']} → {EMOJI[['SELL','HOLD','BUY'].index(r['signal'])]} {r['signal']} "
                f"(conf={r['confidence']:.3f})"
            )
        else:
            print(f"Could not predict for {sym}")
    else:
        predict_all()
