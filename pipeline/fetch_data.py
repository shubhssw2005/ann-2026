"""
Pure Python data fetcher for Lightning AI (no Rust needed).
Fetches top-N coins from Binance futures — same logic as Rust fetcher.
"""

import requests, csv, json, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_FUTURES = "https://fapi.binance.com"
BASE_FUTURES_DATA = "https://fapi.binance.com/futures/data"
DATA_RAW = Path("data/raw")
TOP_N = int(os.getenv("TOP_N", "100"))
INTERVAL = os.getenv("INTERVAL", "15m")
LIMIT = int(os.getenv("LIMIT", "1500"))

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "btc-ann/1.0"})

# ─── Step 1: Get all USDT perpetual symbols ───────────────────────────────────


def get_symbols():
    r = SESSION.get(f"{BASE_FUTURES}/fapi/v1/exchangeInfo", timeout=15)
    r.raise_for_status()
    syms = [
        s["symbol"]
        for s in r.json()["symbols"]
        if s["status"] == "TRADING"
        and s["quoteAsset"] == "USDT"
        and s.get("contractType") == "PERPETUAL"
    ]
    print(f"[fetch] found {len(syms)} active USDT perpetuals")
    return syms


# ─── Step 2: Score by volume + volatility ────────────────────────────────────


def get_tickers():
    r = SESSION.get(f"{BASE_FUTURES}/fapi/v1/ticker/24hr", timeout=15)
    r.raise_for_status()
    return {t["symbol"]: t for t in r.json()}


def score_and_select(symbols, tickers, top_n):
    scored = []
    for sym in symbols:
        t = tickers.get(sym)
        if not t:
            continue
        vol = float(t.get("quoteVolume", 0))
        chg = abs(float(t.get("priceChangePercent", 0)))
        cnt = int(t.get("count", 0))
        if vol < 5_000_000:
            continue
        score = (vol**0.5) * (chg + 0.1) * (cnt**0.3)
        scored.append({"symbol": sym, "vol": vol, "chg": chg, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_n]

    print(f"\n[fetch] Top {len(top)} coins:")
    for i, c in enumerate(top[:10], 1):
        print(
            f"  #{i:3d} {c['symbol']:<16} vol=${c['vol']/1e6:.0f}M  chg={c['chg']:.1f}%"
        )
    print(f"  ... and {len(top)-10} more\n")

    return [c["symbol"] for c in top]


# ─── Step 3: Fetch klines + funding + OI ─────────────────────────────────────


def fetch_klines(symbol):
    url = f"{BASE_FUTURES}/fapi/v1/klines"
    r = SESSION.get(
        url, params={"symbol": symbol, "interval": INTERVAL, "limit": LIMIT}, timeout=15
    )
    r.raise_for_status()
    rows = []
    for k in r.json():
        vol = float(k[5])
        taker_b = float(k[9])
        rows.append(
            {
                "timestamp": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": vol,
                "taker_buy_vol": taker_b,
                "taker_sell_vol": vol - taker_b,
                "num_trades": int(k[8]),
            }
        )
    return rows


def fetch_funding(symbol):
    url = f"{BASE_FUTURES}/fapi/v1/fundingRate"
    r = SESSION.get(url, params={"symbol": symbol, "limit": 1000}, timeout=15)
    r.raise_for_status()
    return [
        {"timestamp": x["fundingTime"], "funding_rate": float(x["fundingRate"])}
        for x in r.json()
    ]


def fetch_oi(symbol):
    url = f"{BASE_FUTURES_DATA}/openInterestHist"
    r = SESSION.get(
        url, params={"symbol": symbol, "period": INTERVAL, "limit": LIMIT}, timeout=15
    )
    r.raise_for_status()
    return [
        {
            "timestamp": x["timestamp"],
            "open_interest": float(x["sumOpenInterest"]),
            "open_interest_value": float(x["sumOpenInterestValue"]),
        }
        for x in r.json()
    ]


def write_csv(rows, path):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def fetch_coin(symbol):
    try:
        klines = fetch_klines(symbol)
        funding = fetch_funding(symbol)
        oi = fetch_oi(symbol)
        write_csv(klines, DATA_RAW / symbol / "klines.csv")
        write_csv(funding, DATA_RAW / symbol / "funding.csv")
        write_csv(oi, DATA_RAW / symbol / "open_interest.csv")
        return symbol, True, len(klines)
    except Exception as e:
        return symbol, False, str(e)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    symbols = get_symbols()
    tickers = get_tickers()
    top = score_and_select(symbols, tickers, TOP_N)

    # Save symbol list
    with open(DATA_RAW / "symbols.json", "w") as f:
        json.dump(top, f)
    print(f"[fetch] saved {len(top)} symbols to data/raw/symbols.json")

    # Fetch all coins in parallel (10 workers — safe for Binance rate limits)
    print(f"[fetch] fetching data for {len(top)} coins...")
    ok = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_coin, sym): sym for sym in top}
        for i, fut in enumerate(as_completed(futures), 1):
            sym, success, info = fut.result()
            if success:
                ok += 1
                print(f"  [{i:3d}/{len(top)}] ✓ {sym:<16} ({info} rows)")
            else:
                fail += 1
                print(f"  [{i:3d}/{len(top)}] ✗ {sym:<16} {info}")

            # Rate limit: small sleep every 20 coins
            if i % 20 == 0:
                time.sleep(0.5)

    print(f"\n[fetch] done: {ok} success, {fail} failed")


if __name__ == "__main__":
    main()
