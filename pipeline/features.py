"""
Feature Engineering Pipeline — Multi-Coin ANN
Processes all top-N coins fetched by Rust, engineers ~50 features each,
runs EDA, builds targets, saves per-coin splits + combined dataset.
"""

import numpy as np
import pandas as pd
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import warnings, joblib, json, traceback

warnings.filterwarnings("ignore")

DATA_RAW = Path("data/raw")
DATA_PROC = Path("data/processed")
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ─── Feature column list ──────────────────────────────────────────────────────

FEATURE_COLS = [
    # Price action
    "return",
    "log_return",
    "hl_ratio",
    "co_ratio",
    "norm_return",
    # Trend
    "ema_2",
    "ema_4",
    "ema_12",
    "ema_24",
    "emsd_2",
    "emsd_4",
    "emsd_12",
    "emsd_24",
    "macd",
    "macd_signal",
    "macd_hist",
    "adx",
    # Volatility
    "bb_upper",
    "bb_lower",
    "bb_pct",
    "bb_width",
    "atr",
    # Momentum
    "rsi_12",
    "rsi_24",
    "rsi_48",
    "stoch_k",
    "stoch_d",
    "williams_r",
    "cci",
    # Volume
    "vol_ratio",
    "obv_norm",
    "vwap_dev",
    "buy_sell_ratio",
    # Market structure
    "funding_rate",
    "oi_change",
    # Time
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Lags
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_4",
    "ret_lag_5",
    # Higher TF
    "daily_trend",
]

# ─── Load symbols ─────────────────────────────────────────────────────────────


def load_symbols() -> list:
    sym_file = DATA_RAW / "symbols.json"
    if not sym_file.exists():
        raise FileNotFoundError("Run Rust fetcher first: cargo run --release")
    with open(sym_file) as f:
        return json.load(f)


# ─── Load single coin ─────────────────────────────────────────────────────────


def load_coin(symbol: str) -> pd.DataFrame | None:
    klines_path = DATA_RAW / symbol / "klines.csv"
    if not klines_path.exists():
        return None

    df = pd.read_csv(klines_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Merge funding rate
    fund_path = DATA_RAW / symbol / "funding.csv"
    if fund_path.exists():
        funding = pd.read_csv(fund_path)
        funding["timestamp"] = pd.to_datetime(funding["timestamp"], unit="ms")
        funding = funding.sort_values("timestamp").drop_duplicates("timestamp")
        df = pd.merge_asof(df, funding, on="timestamp", direction="backward")

    # Merge open interest
    oi_path = DATA_RAW / symbol / "open_interest.csv"
    if oi_path.exists():
        oi = pd.read_csv(oi_path)
        oi["timestamp"] = pd.to_datetime(oi["timestamp"], unit="ms")
        oi = oi.sort_values("timestamp").drop_duplicates("timestamp")
        df = pd.merge_asof(df, oi, on="timestamp", direction="backward")

    df.set_index("timestamp", inplace=True)
    return df


# ─── Feature engineering ──────────────────────────────────────────────────────


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Category 1: Price Action
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_ratio"] = (df["high"] - df["low"]) / df["close"]
    df["co_ratio"] = (df["close"] - df["open"]) / df["open"]
    df["norm_return"] = (df["return"] - df["return"].rolling(24).mean()) / (
        df["return"].rolling(24).std() + 1e-9
    )

    # Category 2: Trend
    for w in [2, 4, 12, 24]:
        df[f"ema_{w}"] = df["close"].ewm(span=w).mean() / df["close"] - 1
        df[f"emsd_{w}"] = df["log_return"].ewm(span=w).std()

    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = (ema12 - ema26) / df["close"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14) / 100

    # Category 3: Volatility
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = (sma20 + 2 * std20) / df["close"] - 1
    df["bb_lower"] = (sma20 - 2 * std20) / df["close"] - 1
    df["bb_pct"] = (df["close"] - (sma20 - 2 * std20)) / (4 * std20 + 1e-9)
    df["bb_width"] = 4 * std20 / (sma20 + 1e-9)
    df["atr"] = (
        ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        / df["close"]
    )

    # Category 4: Momentum
    for w in [12, 24, 48]:
        df[f"rsi_{w}"] = ta.momentum.rsi(df["close"], window=w) / 100
    df["stoch_k"] = ta.momentum.stoch(df["high"], df["low"], df["close"]) / 100
    df["stoch_d"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"]) / 100
    df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"]) / 100
    df["cci"] = (
        ta.trend.cci(df["high"], df["low"], df["close"], window=20) / 200
    ).clip(-1, 1)

    # Category 5: Volume
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(24).mean() + 1e-9)
    obv = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["obv_norm"] = (obv - obv.rolling(48).mean()) / (obv.rolling(48).std() + 1e-9)
    vwap = (df["close"] * df["volume"]).rolling(24).sum() / (
        df["volume"].rolling(24).sum() + 1e-9
    )
    df["vwap_dev"] = (df["close"] - vwap) / (vwap + 1e-9)

    if "taker_buy_vol" in df.columns and "taker_sell_vol" in df.columns:
        total = df["taker_buy_vol"] + df["taker_sell_vol"] + 1e-9
        df["buy_sell_ratio"] = df["taker_buy_vol"] / total
    else:
        df["buy_sell_ratio"] = 0.5

    # Category 6: Market Structure
    if "funding_rate" in df.columns:
        df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
    else:
        df["funding_rate"] = 0.0

    if "open_interest" in df.columns:
        df["oi_change"] = df["open_interest"].pct_change().clip(-0.5, 0.5)
    else:
        df["oi_change"] = 0.0

    # Category 7: Time Encoding
    hour = df.index.hour
    dow = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Category 8: Lagged Features
    for lag in range(1, 6):
        df[f"ret_lag_{lag}"] = df["norm_return"].shift(lag)

    # Higher TF daily trend
    daily_close = df["close"].resample("1D").last()
    daily_ema20 = daily_close.ewm(span=20).mean()
    daily_ema20 = daily_ema20.reindex(df.index, method="ffill")
    df["daily_trend"] = (df["close"] / (daily_ema20 + 1e-9)) - 1

    return df


# ─── Target ───────────────────────────────────────────────────────────────────


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    future_return = df["close"].shift(-1) / df["close"] - 1
    q_low = future_return.rolling(200, min_periods=50).quantile(0.33)
    q_high = future_return.rolling(200, min_periods=50).quantile(0.67)
    df["target"] = 1
    df.loc[future_return <= q_low, "target"] = 0  # SELL
    df.loc[future_return >= q_high, "target"] = 2  # BUY
    return df


# ─── Per-coin quality check ───────────────────────────────────────────────────


def quality_check(df: pd.DataFrame, symbol: str) -> bool:
    """Returns True if coin has enough clean data to be useful."""
    if len(df) < 300:
        print(f"  [skip] {symbol}: too few rows ({len(df)})")
        return False
    nan_pct = df[FEATURE_COLS].isnull().mean().mean()
    if nan_pct > 0.3:
        print(f"  [skip] {symbol}: too many NaNs ({nan_pct:.1%})")
        return False
    return True


# ─── Process single coin ─────────────────────────────────────────────────────


def process_coin(symbol: str) -> tuple | None:
    try:
        df = load_coin(symbol)
        if df is None or len(df) < 200:
            return None

        df = build_features(df)
        df = build_target(df)

        available = [c for c in FEATURE_COLS if c in df.columns]
        df_clean = df.dropna(subset=available + ["target"]).copy()

        if not quality_check(df_clean, symbol):
            return None

        # Clip outliers
        for col in available:
            mu, sigma = df_clean[col].mean(), df_clean[col].std()
            df_clean[col] = df_clean[col].clip(mu - 3 * sigma, mu + 3 * sigma)

        # Add symbol column for combined dataset
        df_clean["symbol"] = symbol

        return df_clean, available

    except Exception as e:
        print(f"  [error] {symbol}: {e}")
        return None


# ─── EDA on combined dataset ──────────────────────────────────────────────────


def run_combined_eda(combined: pd.DataFrame, feature_cols: list):
    print("\n=== Combined EDA ===")
    print(f"Total rows: {len(combined):,}  Coins: {combined['symbol'].nunique()}")
    print(f"Date range: {combined.index.min()} → {combined.index.max()}")

    # Rows per coin
    coin_counts = combined["symbol"].value_counts()
    print(f"\nRows per coin (top 10):\n{coin_counts.head(10)}")

    # Target distribution
    print(
        f"\nTarget distribution:\n{combined['target'].value_counts(normalize=True).round(3)}"
    )

    # Correlation heatmap (sample for speed)
    sample = combined[feature_cols].sample(min(5000, len(combined)), random_state=42)
    corr = sample.corr()
    plt.figure(figsize=(22, 18))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        linewidths=0.2,
        annot=False,
    )
    plt.title("Feature Correlation Matrix (all coins combined)")
    plt.tight_layout()
    plt.savefig(DATA_PROC / "correlation_matrix.png", dpi=100)
    plt.close()

    # Volume distribution across coins
    plt.figure(figsize=(14, 5))
    coin_counts.head(50).plot(kind="bar")
    plt.title("Rows per Coin (top 50)")
    plt.tight_layout()
    plt.savefig(DATA_PROC / "rows_per_coin.png", dpi=100)
    plt.close()

    print("[EDA] plots saved to data/processed/")


# ─── Feature importance ───────────────────────────────────────────────────────


def feature_importance_report(X: np.ndarray, y: np.ndarray, feature_cols: list):
    print("\n[Feature Importance] computing mutual info...")
    mi = mutual_info_classif(X[:10000], y[:10000], random_state=42)  # cap for speed
    fi = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
    print(fi.to_string())

    plt.figure(figsize=(10, 14))
    fi.plot(kind="barh")
    plt.title("Feature Importance (Mutual Information)")
    plt.tight_layout()
    plt.savefig(DATA_PROC / "feature_importance.png", dpi=100)
    plt.close()
    return fi


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    symbols = load_symbols()
    print(f"Processing {len(symbols)} coins...")

    all_dfs = []
    all_features = set(FEATURE_COLS)
    processed = []
    skipped = []

    for i, sym in enumerate(symbols):
        print(f"[{i+1:3d}/{len(symbols)}] {sym}", end=" ... ")
        result = process_coin(sym)
        if result is None:
            skipped.append(sym)
            print("skipped")
            continue
        df_clean, available = result
        all_dfs.append(df_clean)
        all_features &= set(available)
        processed.append(sym)
        print(f"ok ({len(df_clean)} rows)")

    print(f"\nProcessed: {len(processed)}  Skipped: {len(skipped)}")
    print(f"Common features across all coins: {len(all_features)}")

    if not all_dfs:
        raise RuntimeError("No coins processed successfully.")

    # Combine all coins
    feature_cols = [c for c in FEATURE_COLS if c in all_features]
    combined = pd.concat(all_dfs, axis=0).sort_index()

    run_combined_eda(combined, feature_cols)

    # Time-ordered split (global, preserves temporal order)
    n = len(combined)
    s1 = int(n * 0.70)
    s2 = int(n * 0.85)

    X = combined[feature_cols].values.astype(np.float32)
    y = combined["target"].values.astype(np.int64)

    X_train, y_train = X[:s1], y[:s1]
    X_val, y_val = X[s1:s2], y[s1:s2]
    X_test, y_test = X[s2:], y[s2:]

    # Scaler fit only on train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, "models/scaler.pkl")

    feature_importance_report(X_train, y_train, feature_cols)

    # Save
    np.save(DATA_PROC / "X_train.npy", X_train)
    np.save(DATA_PROC / "y_train.npy", y_train)
    np.save(DATA_PROC / "X_val.npy", X_val)
    np.save(DATA_PROC / "y_val.npy", y_val)
    np.save(DATA_PROC / "X_test.npy", X_test)
    np.save(DATA_PROC / "y_test.npy", y_test)

    with open(DATA_PROC / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    with open(DATA_PROC / "processed_coins.json", "w") as f:
        json.dump(processed, f)

    print(
        f"\n[done] {len(feature_cols)} features | "
        f"train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}"
    )
    print(f"Coins in dataset: {len(processed)}")


if __name__ == "__main__":
    main()
