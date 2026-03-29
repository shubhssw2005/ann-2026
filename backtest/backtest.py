"""
Backtester — Multi-Coin ANN Signal Backtest
- Simulates trading on test set using model signals
- Per-coin + aggregate P&L, Sharpe, max drawdown, win rate
- Outputs JSON for frontend dashboard
"""

import numpy as np
import pandas as pd
import torch
import joblib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from model import BTCANN
from features import load_coin, build_features, FEATURE_COLS

DEVICE = torch.device("cpu")
LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}
DATA_PROC = Path("data/processed")
OUT_DIR = Path("data/backtest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────

CFG = {
    "fee": 0.0004,  # 0.04% taker fee per side
    "slippage": 0.0002,  # 0.02% slippage
    "position_size": 1.0,  # fraction of capital per trade
    "stop_loss": -0.03,  # -3% stop loss
    "take_profit": 0.05,  # +5% take profit
    "test_split": 0.85,  # same as training split
}

# ─── Load model ───────────────────────────────────────────────────────────────


def load_model():
    ckpt = torch.load("models/best_model.pt", map_location=DEVICE)
    model = BTCANN(n_features=ckpt["n_features"], n_classes=3).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = joblib.load("models/scaler.pkl")
    with open(DATA_PROC / "feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols


# ─── Metrics ──────────────────────────────────────────────────────────────────


def sharpe(returns: np.ndarray, periods_per_year: int = 35040) -> float:
    """Annualized Sharpe (15m candles = 35040/year)."""
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-9)
    return float(dd.min())


def win_rate(pnls: list) -> float:
    if not pnls:
        return 0.0
    return sum(1 for p in pnls if p > 0) / len(pnls)


# ─── Backtest single coin ─────────────────────────────────────────────────────


def backtest_coin(symbol: str, model, scaler, feature_cols) -> dict | None:
    df = load_coin(symbol)
    if df is None or len(df) < 200:
        return None

    from features import build_target

    df = build_features(df)
    df = build_target(df)

    available = [c for c in feature_cols if c in df.columns]
    df_clean = df.dropna(subset=available + ["target"]).copy()
    if len(df_clean) < 100:
        return None

    # Use test portion only (last 15%)
    n = len(df_clean)
    start = int(n * CFG["test_split"])
    df_test = df_clean.iloc[start:].copy()
    if len(df_test) < 20:
        return None

    # Clip + scale
    for col in available:
        mu, sigma = df_clean[col].mean(), df_clean[col].std()
        df_test[col] = df_test[col].clip(mu - 3 * sigma, mu + 3 * sigma)

    X = scaler.transform(df_test[available].values.astype(np.float32))
    with torch.no_grad():
        probs = torch.softmax(
            model(torch.tensor(X, dtype=torch.float32)), dim=-1
        ).numpy()

    signals = probs.argmax(1)  # 0=SELL 1=HOLD 2=BUY
    confidence = probs.max(1)

    closes = df_test["close"].values
    timestamps = df_test.index.astype(str).tolist()

    # ── Simulate trades ───────────────────────────────────────────────────────
    capital = 1.0
    equity = [capital]
    trade_pnls = []
    trades = []
    position = None  # {"side": "long"/"short", "entry": float, "entry_idx": int}

    cost = CFG["fee"] + CFG["slippage"]

    for i in range(len(signals) - 1):
        sig = signals[i]
        conf = confidence[i]
        price_now = closes[i]
        price_next = closes[i + 1]

        # Close existing position
        if position is not None:
            ret = (price_now - position["entry"]) / position["entry"]
            if position["side"] == "short":
                ret = -ret

            # Stop loss / take profit check
            hit_sl = ret <= CFG["stop_loss"]
            hit_tp = ret >= CFG["take_profit"]

            if (
                hit_sl
                or hit_tp
                or (position["side"] == "long" and sig != 2)
                or (position["side"] == "short" and sig != 0)
            ):
                net_ret = ret - 2 * cost
                capital *= 1 + net_ret * CFG["position_size"]
                trade_pnls.append(net_ret)
                trades.append(
                    {
                        "symbol": symbol,
                        "side": position["side"],
                        "entry": float(position["entry"]),
                        "exit": float(price_now),
                        "entry_ts": timestamps[position["entry_idx"]],
                        "exit_ts": timestamps[i],
                        "pnl": round(net_ret * 100, 4),
                        "hit_sl": hit_sl,
                        "hit_tp": hit_tp,
                    }
                )
                position = None

        # Open new position (only on high confidence)
        if position is None and conf > 0.45:
            if sig == 2:  # BUY → long
                position = {"side": "long", "entry": price_next, "entry_idx": i + 1}
            elif sig == 0:  # SELL → short
                position = {"side": "short", "entry": price_next, "entry_idx": i + 1}

        equity.append(capital)

    equity_arr = np.array(equity)
    returns = np.diff(equity_arr) / equity_arr[:-1]

    result = {
        "symbol": symbol,
        "n_trades": len(trade_pnls),
        "win_rate": round(win_rate(trade_pnls) * 100, 2),
        "total_return": round((capital - 1.0) * 100, 4),
        "sharpe": round(sharpe(returns), 3),
        "max_drawdown": round(max_drawdown(equity_arr) * 100, 4),
        "final_capital": round(capital, 6),
        "equity_curve": [round(e, 6) for e in equity_arr[::4].tolist()],  # downsample
        "timestamps": timestamps[::4],
        "trades": trades[-50:],  # last 50 trades for display
        "signals": [LABELS[int(s)] for s in signals],
        "closes": [round(float(c), 4) for c in closes[::4].tolist()],
        "confidence": [round(float(c), 3) for c in confidence[::4].tolist()],
    }
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    model, scaler, feature_cols = load_model()

    with open(DATA_PROC / "processed_coins.json") as f:
        coins = json.load(f)

    print(f"Backtesting {len(coins)} coins on test set (last 15% of data)...\n")

    results = []
    failed = []

    for i, sym in enumerate(coins):
        print(f"[{i+1:3d}/{len(coins)}] {sym}", end=" ... ")
        try:
            r = backtest_coin(sym, model, scaler, feature_cols)
            if r:
                results.append(r)
                print(
                    f"return={r['total_return']:+.2f}%  sharpe={r['sharpe']:.2f}  "
                    f"trades={r['n_trades']}  wr={r['win_rate']:.1f}%"
                )
            else:
                failed.append(sym)
                print("skipped")
        except Exception as e:
            failed.append(sym)
            print(f"error: {e}")

    if not results:
        print("No results.")
        return

    # ── Aggregate stats ───────────────────────────────────────────────────────
    returns = [r["total_return"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    drawdowns = [r["max_drawdown"] for r in results]
    win_rates = [r["win_rate"] for r in results]

    summary = {
        "n_coins": len(results),
        "avg_return": round(float(np.mean(returns)), 4),
        "median_return": round(float(np.median(returns)), 4),
        "avg_sharpe": round(float(np.mean(sharpes)), 3),
        "avg_max_drawdown": round(float(np.mean(drawdowns)), 4),
        "avg_win_rate": round(float(np.mean(win_rates)), 2),
        "profitable_coins": sum(1 for r in returns if r > 0),
        "best_coin": max(results, key=lambda x: x["total_return"])["symbol"],
        "worst_coin": min(results, key=lambda x: x["total_return"])["symbol"],
        "best_sharpe_coin": max(results, key=lambda x: x["sharpe"])["symbol"],
    }

    print(f"\n{'='*50}")
    print(f"BACKTEST SUMMARY ({len(results)} coins)")
    print(f"{'='*50}")
    for k, v in summary.items():
        print(f"  {k:<25} {v}")

    # Sort by Sharpe for display
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    output = {"summary": summary, "coins": results}
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
