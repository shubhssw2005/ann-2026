/// WebSocket Stream — Multi-Coin Real-Time Kline Listener
///
/// Subscribes to <symbol>@kline_<interval> for all top-N coins simultaneously
/// using Binance combined stream endpoint.
/// On candle close (kline.x == true), appends the new row to data/raw/<SYMBOL>/klines.csv
/// and writes a trigger file so Python predict.py fires instantly.
///
/// Binance combined stream limit: 1024 streams per connection.
/// We batch into multiple connections if needed (100 coins × 1 stream = 100, well within limit).

use anyhow::{Result, Context};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::Path;
use std::process::Command;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn, error};

const WS_BASE: &str = "wss://fstream.binance.com/stream?streams=";
const MAX_STREAMS_PER_CONN: usize = 200; // safe limit per connection

// ─── Kline event from WebSocket ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct WsEnvelope {
    stream: String,
    data: WsKlineEvent,
}

#[derive(Debug, Deserialize)]
struct WsKlineEvent {
    #[serde(rename = "E")]
    event_time: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "k")]
    kline: WsKline,
}

#[derive(Debug, Deserialize)]
struct WsKline {
    #[serde(rename = "t")]
    open_time: i64,
    #[serde(rename = "T")]
    close_time: i64,
    #[serde(rename = "o")]
    open: String,
    #[serde(rename = "h")]
    high: String,
    #[serde(rename = "l")]
    low: String,
    #[serde(rename = "c")]
    close: String,
    #[serde(rename = "v")]
    volume: String,
    #[serde(rename = "n")]
    num_trades: u64,
    #[serde(rename = "x")]
    is_closed: bool,          // TRUE = candle closed → trigger inference
    #[serde(rename = "V")]
    taker_buy_base_vol: String,
    #[serde(rename = "Q")]
    taker_buy_quote_vol: String,
}

#[derive(Debug, Serialize)]
struct KlineRow {
    timestamp: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    taker_buy_vol: f64,
    taker_sell_vol: f64,
    num_trades: u64,
}

// ─── Closed candle handler ────────────────────────────────────────────────────

fn handle_closed_candle(symbol: &str, k: &WsKline) {
    let vol: f64     = k.volume.parse().unwrap_or(0.0);
    let tbv: f64     = k.taker_buy_base_vol.parse().unwrap_or(0.0);
    let row = KlineRow {
        timestamp:     k.open_time,
        open:          k.open.parse().unwrap_or(0.0),
        high:          k.high.parse().unwrap_or(0.0),
        low:           k.low.parse().unwrap_or(0.0),
        close:         k.close.parse().unwrap_or(0.0),
        volume:        vol,
        taker_buy_vol: tbv,
        taker_sell_vol: vol - tbv,
        num_trades:    k.num_trades,
    };

    let path = format!("data/raw/{}/klines.csv", symbol);
    let file_exists = Path::new(&path).exists();

    // Append row to CSV (create with header if new)
    match csv::WriterBuilder::new()
        .has_headers(!file_exists)
        .flexible(true)
        .from_path(&path)
    {
        Ok(mut wtr) => {
            if let Err(e) = wtr.serialize(&row) {
                warn!("[{}] csv write error: {}", symbol, e);
            }
        }
        Err(e) => warn!("[{}] csv open error: {}", symbol, e),
    }

    // Write trigger file → Python scheduler watches this
    let trigger = format!("data/raw/{}/candle_closed", symbol);
    let _ = std::fs::write(&trigger, k.close_time.to_string());

    info!("[CLOSED] {} close={} vol={:.1}", symbol, row.close, vol);

    // Spawn Python predict for this coin (non-blocking)
    let sym = symbol.to_string();
    tokio::spawn(async move {
        let output = Command::new("python")
            .args(["pipeline/predict.py", &sym])
            .output();
        match output {
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                if !stdout.trim().is_empty() {
                    info!("[predict] {}", stdout.trim());
                }
            }
            Err(e) => warn!("[predict] spawn error: {}", e),
        }
    });
}

// ─── Single WebSocket connection ──────────────────────────────────────────────

async fn run_ws_connection(streams: Vec<String>, tx: mpsc::Sender<String>) -> Result<()> {
    let stream_str = streams.join("/");
    let url = format!("{}{}", WS_BASE, stream_str);

    info!("Connecting to {} streams...", streams.len());

    let (ws_stream, _) = connect_async(&url).await
        .context("WebSocket connect failed")?;

    let (mut write, mut read) = ws_stream.split();

    // Binance requires pong response to ping
    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                let _ = tx.send(text).await;
            }
            Ok(Message::Ping(data)) => {
                let _ = write.send(Message::Pong(data)).await;
            }
            Ok(Message::Close(_)) => {
                warn!("WebSocket closed by server, will reconnect...");
                break;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
    Ok(())
}

// ─── Message processor ────────────────────────────────────────────────────────

async fn process_messages(mut rx: mpsc::Receiver<String>) {
    while let Some(text) = rx.recv().await {
        match serde_json::from_str::<WsEnvelope>(&text) {
            Ok(env) => {
                if env.data.kline.is_closed {
                    handle_closed_candle(&env.data.symbol, &env.data.kline);
                }
            }
            Err(_) => {
                // Some messages are not kline events (e.g. subscription confirmations)
            }
        }
    }
}

// ─── Load symbols ─────────────────────────────────────────────────────────────

fn load_symbols() -> Result<Vec<String>> {
    let data = std::fs::read_to_string("data/raw/symbols.json")
        .context("Run btc-fetcher first to generate symbols.json")?;
    let syms: Vec<String> = serde_json::from_str(&data)?;
    Ok(syms)
}

// ─── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    dotenv::dotenv().ok();

    let interval = env::var("INTERVAL").unwrap_or_else(|_| "15m".to_string());
    let symbols  = load_symbols()?;

    info!("Starting WebSocket streams for {} coins @ {}...", symbols.len(), interval);

    // Build stream names: btcusdt@kline_15m
    let stream_names: Vec<String> = symbols.iter()
        .map(|s| format!("{}@kline_{}", s.to_lowercase(), interval))
        .collect();

    // Split into batches (max MAX_STREAMS_PER_CONN per connection)
    let batches: Vec<Vec<String>> = stream_names
        .chunks(MAX_STREAMS_PER_CONN)
        .map(|c| c.to_vec())
        .collect();

    info!("Using {} WebSocket connections", batches.len());

    let (tx, rx) = mpsc::channel::<String>(10_000);

    // Spawn message processor
    tokio::spawn(process_messages(rx));

    // Spawn one task per connection batch, with auto-reconnect
    let mut handles = vec![];
    for batch in batches {
        let tx_clone = tx.clone();
        handles.push(tokio::spawn(async move {
            loop {
                if let Err(e) = run_ws_connection(batch.clone(), tx_clone.clone()).await {
                    error!("WS connection error: {} — reconnecting in 5s", e);
                }
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        }));
    }

    // Wait forever
    for h in handles {
        let _ = h.await;
    }

    Ok(())
}
