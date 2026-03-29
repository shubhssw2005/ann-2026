"""
Scheduler — Event-driven, WebSocket-based.

Two modes:
  1. ws   (default) — watches trigger files written by ws-stream Rust binary.
                      Fires predict instantly on candle close. Zero latency.
  2. poll            — fallback REST polling every candle close (no WebSocket).

Usage:
  python scripts/scheduler.py ws      # WebSocket mode (recommended)
  python scripts/scheduler.py poll    # REST polling fallback
"""

import subprocess, time, os, sys, json
from datetime import datetime, timezone
from pathlib import Path
import threading

ROOT = Path(__file__).parent.parent
INTERVAL_MINUTES = int(os.getenv("INTERVAL_MINUTES", "15"))
RETRAIN_EVERY = int(os.getenv("RETRAIN_EVERY", "96"))
DATA_RAW = ROOT / "data" / "raw"


def get_bin_dir() -> str:
    """Detect cargo target dir (handles CARGO_TARGET_DIR overrides)."""
    try:
        import subprocess, json

        result = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        target = json.loads(result.stdout)["target_directory"]
        return str(Path(target) / "release")
    except Exception:
        return str(ROOT / "target" / "release")


candle_count = 0
retrain_lock = threading.Lock()

# ─── Helpers ──────────────────────────────────────────────────────────────────


def run(cmd: list, cwd=None):
    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] $ {' '.join(str(c) for c in cmd)}"
    )
    subprocess.run(cmd, cwd=str(cwd or ROOT))


def maybe_retrain():
    global candle_count
    candle_count += 1
    if candle_count % RETRAIN_EVERY == 1:
        with retrain_lock:
            print(f"[scheduler] retraining on all coins (candle #{candle_count})")
            run(["python", "pipeline/features.py"])
            run(["python", "pipeline/train.py"])


# ─── Mode 1: WebSocket-driven ─────────────────────────────────────────────────


def watch_triggers():
    """
    Watches data/raw/<SYMBOL>/candle_closed trigger files.
    ws-stream Rust binary writes these on every closed candle.
    We batch-collect all coins that closed in the same interval window
    and run a single predict pass.
    """
    print(f"[ws-mode] watching trigger files in {DATA_RAW}")

    sym_file = DATA_RAW / "symbols.json"
    if not sym_file.exists():
        print("[error] symbols.json not found — run btc-fetcher first")
        sys.exit(1)

    with open(sym_file) as f:
        symbols = json.load(f)

    trigger_files = {sym: DATA_RAW / sym / "candle_closed" for sym in symbols}
    last_seen = {sym: None for sym in symbols}

    print(f"[ws-mode] monitoring {len(symbols)} coins @ {INTERVAL_MINUTES}m interval")
    print("[ws-mode] make sure ws-stream binary is running in another terminal\n")

    while True:
        closed_now = []

        for sym, tfile in trigger_files.items():
            if not tfile.exists():
                continue
            try:
                mtime = tfile.stat().st_mtime
                if mtime != last_seen[sym]:
                    last_seen[sym] = mtime
                    closed_now.append(sym)
            except FileNotFoundError:
                pass

        if closed_now:
            print(
                f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                f"Candle closed: {len(closed_now)} coins → running predict"
            )
            run(["python", "pipeline/predict.py"])
            maybe_retrain()

        time.sleep(0.5)  # poll trigger files every 500ms


# ─── Mode 2: REST polling fallback ────────────────────────────────────────────


def seconds_to_next_candle(interval_min: int) -> float:
    now = datetime.now(timezone.utc)
    elapsed = (now.minute % interval_min) * 60 + now.second
    return interval_min * 60 - elapsed + 2


def poll_mode():
    print(f"[poll-mode] interval={INTERVAL_MINUTES}m  retrain_every={RETRAIN_EVERY}")
    while True:
        wait = seconds_to_next_candle(INTERVAL_MINUTES)
        print(f"[poll-mode] sleeping {wait:.0f}s until next candle close...")
        time.sleep(wait)

        run([f"{BIN_DIR}/btc-fetcher"])
        run(["python", "pipeline/features.py"])
        maybe_retrain()
        run(["python", "pipeline/predict.py"])


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ws"

    if mode == "poll":
        poll_mode()
    else:
        watch_triggers()
