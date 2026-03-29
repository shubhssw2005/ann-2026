use anyhow::{Result, Context};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::collections::HashMap;
use tracing::{info, warn};

const BASE_SPOT: &str = "https://api.binance.com";
const BASE_FUTURES: &str = "https://fapi.binance.com";
const BASE_FUTURES_DATA: &str = "https://fapi.binance.com/futures/data";

// ─── Exchange Info ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ExchangeInfo {
    symbols: Vec<SymbolInfo>,
}

#[derive(Debug, Deserialize)]
struct SymbolInfo {
    symbol: String,
    status: String,
    #[serde(rename = "quoteAsset")]
    quote_asset: String,
    #[serde(rename = "contractType")]
    contract_type: Option<String>,
}

// ─── 24hr Ticker ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone)]
struct Ticker24h {
    symbol: String,
    #[serde(rename = "quoteVolume")]
    quote_volume: String,
    #[serde(rename = "priceChangePercent")]
    price_change_pct: String,
    #[serde(rename = "count")]
    trade_count: Option<u64>,
}

// ─── Kline ────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct KlineRaw(
    i64, String, String, String, String, String,
    i64, String, u64, String, String, String,
);

#[derive(Debug, Serialize)]
struct Kline {
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

// ─── Funding / OI ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct FundingRate {
    #[serde(rename = "fundingTime")]
    funding_time: i64,
    #[serde(rename = "fundingRate")]
    funding_rate: String,
}

#[derive(Debug, Serialize)]
struct FundingOut {
    timestamp: i64,
    funding_rate: f64,
}

#[derive(Debug, Deserialize)]
struct OpenInterestHist {
    timestamp: i64,
    #[serde(rename = "sumOpenInterest")]
    sum_open_interest: String,
    #[serde(rename = "sumOpenInterestValue")]
    sum_open_interest_value: String,
}

#[derive(Debug, Serialize)]
struct OIOut {
    timestamp: i64,
    open_interest: f64,
    open_interest_value: f64,
}

// ─── Coin Score ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
struct CoinScore {
    symbol: String,
    quote_volume_24h: f64,
    trade_count: u64,
    volatility_pct: f64,
    score: f64,
}

// ─── Fetchers ─────────────────────────────────────────────────────────────────

async fn get_futures_symbols(client: &Client) -> Result<Vec<String>> {
    let url = format!("{}/fapi/v1/exchangeInfo", BASE_FUTURES);
    let info: ExchangeInfo = client.get(&url).send().await?.json().await
        .context("Failed to get futures exchange info")?;

    let symbols: Vec<String> = info.symbols.iter()
        .filter(|s| s.status == "TRADING"
            && s.quote_asset == "USDT"
            && s.contract_type.as_deref() == Some("PERPETUAL"))
        .map(|s| s.symbol.clone())
        .collect();

    info!("Found {} active USDT perpetual futures", symbols.len());
    Ok(symbols)
}

async fn get_24h_tickers(client: &Client) -> Result<HashMap<String, Ticker24h>> {
    let url = format!("{}/fapi/v1/ticker/24hr", BASE_FUTURES);
    let tickers: Vec<Ticker24h> = client.get(&url).send().await?.json().await
        .context("Failed to get 24h tickers")?;

    Ok(tickers.into_iter().map(|t| (t.symbol.clone(), t)).collect())
}

fn score_coins(symbols: &[String], tickers: &HashMap<String, Ticker24h>, top_n: usize) -> Vec<CoinScore> {
    let mut scores: Vec<CoinScore> = symbols.iter().filter_map(|sym| {
        let t = tickers.get(sym)?;
        let vol: f64 = t.quote_volume.parse().unwrap_or(0.0);
        let chg: f64 = t.price_change_pct.parse::<f64>().unwrap_or(0.0).abs();
        let cnt: u64 = t.trade_count.unwrap_or(0);

        // Skip very low volume / illiquid coins
        if vol < 5_000_000.0 { return None; }

        // Score = log(volume) * volatility * log(trade_count+1)
        // High volume + high volatility + high activity = best for ANN
        let score = (vol.ln()) * (chg + 0.1) * ((cnt as f64 + 1.0).ln());

        Some(CoinScore {
            symbol: sym.clone(),
            quote_volume_24h: vol,
            trade_count: cnt,
            volatility_pct: chg,
            score,
        })
    }).collect();

    scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    scores.truncate(top_n);
    scores
}

async fn fetch_klines(client: &Client, symbol: &str, interval: &str, limit: usize) -> Result<Vec<Kline>> {
    let url = format!(
        "{}/fapi/v1/klines?symbol={}&interval={}&limit={}",
        BASE_FUTURES, symbol, interval, limit
    );
    let raw: Vec<KlineRaw> = client.get(&url).send().await?.json().await
        .context(format!("klines failed for {}", symbol))?;

    Ok(raw.into_iter().map(|k| {
        let vol: f64 = k.5.parse().unwrap_or(0.0);
        let tbv: f64 = k.9.parse().unwrap_or(0.0);
        Kline {
            timestamp: k.0,
            open: k.1.parse().unwrap_or(0.0),
            high: k.2.parse().unwrap_or(0.0),
            low: k.3.parse().unwrap_or(0.0),
            close: k.4.parse().unwrap_or(0.0),
            volume: vol,
            taker_buy_vol: tbv,
            taker_sell_vol: vol - tbv,
            num_trades: k.8,
        }
    }).collect())
}

async fn fetch_funding(client: &Client, symbol: &str, limit: usize) -> Result<Vec<FundingOut>> {
    let url = format!(
        "{}/fapi/v1/fundingRate?symbol={}&limit={}",
        BASE_FUTURES, symbol, limit
    );
    let raw: Vec<FundingRate> = client.get(&url).send().await?.json().await
        .context(format!("funding failed for {}", symbol))?;

    Ok(raw.into_iter().map(|f| FundingOut {
        timestamp: f.funding_time,
        funding_rate: f.funding_rate.parse().unwrap_or(0.0),
    }).collect())
}

async fn fetch_oi(client: &Client, symbol: &str, period: &str, limit: usize) -> Result<Vec<OIOut>> {
    let url = format!(
        "{}/openInterestHist?symbol={}&period={}&limit={}",
        BASE_FUTURES_DATA, symbol, period, limit
    );
    let raw: Vec<OpenInterestHist> = client.get(&url).send().await?.json().await
        .context(format!("OI failed for {}", symbol))?;

    Ok(raw.into_iter().map(|o| OIOut {
        timestamp: o.timestamp,
        open_interest: o.sum_open_interest.parse().unwrap_or(0.0),
        open_interest_value: o.sum_open_interest_value.parse().unwrap_or(0.0),
    }).collect())
}

// ─── Writers ──────────────────────────────────────────────────────────────────

fn write_csv<T: Serialize>(data: &[T], path: &str) -> Result<()> {
    std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap())?;
    let mut wtr = csv::Writer::from_path(path)?;
    for row in data { wtr.serialize(row)?; }
    wtr.flush()?;
    Ok(())
}

// ─── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    dotenv::dotenv().ok();

    let interval = env::var("INTERVAL").unwrap_or_else(|_| "15m".to_string());
    let limit: usize = env::var("LIMIT").unwrap_or_else(|_| "1500".to_string()).parse().unwrap_or(1500);
    let top_n: usize = env::var("TOP_N").unwrap_or_else(|_| "100".to_string()).parse().unwrap_or(100);

    info!("Scanning all Binance USDT perpetual futures...");

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    // Step 1: Get all futures symbols
    let symbols = get_futures_symbols(&client).await?;

    // Step 2: Score all coins by volume + volatility + activity
    let tickers = get_24h_tickers(&client).await?;
    let top_coins = score_coins(&symbols, &tickers, top_n);

    info!("Top {} coins selected:", top_coins.len());
    for (i, c) in top_coins.iter().enumerate() {
        info!("  #{:3} {:12} vol=${:.0}M  vol%={:.1}%  score={:.1}",
            i+1, c.symbol, c.quote_volume_24h/1e6, c.volatility_pct, c.score);
    }

    // Save coin list
    write_csv(&top_coins, "data/raw/top_coins.csv")?;

    // Save symbol list for Python
    let symbol_list: Vec<String> = top_coins.iter().map(|c| c.symbol.clone()).collect();
    std::fs::write("data/raw/symbols.json", serde_json::to_string_pretty(&symbol_list)?)?;
    info!("Saved {} symbols to data/raw/symbols.json", symbol_list.len());

    // Step 3: Fetch data for each coin (rate-limited, batched)
    let batch_size = 10usize;
    let mut success = 0usize;
    let mut failed  = 0usize;

    for chunk in top_coins.chunks(batch_size) {
        let mut handles = vec![];

        for coin in chunk {
            let sym      = coin.symbol.clone();
            let client_c = client.clone();
            let interval_c = interval.clone();

            handles.push(tokio::spawn(async move {
                let klines  = fetch_klines(&client_c, &sym, &interval_c, limit).await;
                let funding = fetch_funding(&client_c, &sym, 1000).await;
                let oi      = fetch_oi(&client_c, &sym, &interval_c, limit).await;
                (sym, klines, funding, oi)
            }));
        }

        for handle in handles {
            match handle.await {
                Ok((sym, klines_r, funding_r, oi_r)) => {
                    let mut ok = true;
                    if let Ok(k) = klines_r {
                        if let Err(e) = write_csv(&k, &format!("data/raw/{}/klines.csv", sym)) {
                            warn!("{} klines write error: {}", sym, e); ok = false;
                        }
                    } else { warn!("{} klines fetch failed", sym); ok = false; }

                    if let Ok(f) = funding_r {
                        let _ = write_csv(&f, &format!("data/raw/{}/funding.csv", sym));
                    }
                    if let Ok(o) = oi_r {
                        let _ = write_csv(&o, &format!("data/raw/{}/open_interest.csv", sym));
                    }

                    if ok { success += 1; info!("✓ {}", sym); }
                    else  { failed  += 1; }
                }
                Err(e) => { warn!("Task panicked: {}", e); failed += 1; }
            }
        }

        // Rate limit: 10 req/s safe margin
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    info!("Fetch complete: {} success, {} failed", success, failed);
    Ok(())
}
