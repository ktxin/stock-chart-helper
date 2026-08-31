"""
Build a static, pre-screened list of large US stocks for the app's
ticker-picker dropdown.

Two-stage funnel to keep this affordable against Yahoo Finance's rate
limits:
  Stage 1 (cheap): fast_info market cap for every candidate ticker.
                    Drops anything under the market-cap bar immediately
                    without the expensive full .info call. Checkpointed
                    to disk so Stage 2 failures never require redoing it.
  Stage 2 (expensive): full .info for stage-1 survivors only, to get
                    the precise floatShares figure and company name.

Both stages process in small chunks with a pause between chunks (steady
drip instead of a burst -- rate limiters tolerate this far better) and
retry individual rate-limit errors with backoff. Starts with a cooldown
sleep in case a prior run already tripped Yahoo's limiter.

Output: us_stocks_screened.json -- [{ticker, name, market_cap, float_shares}, ...]
sorted by market cap descending. Re-run periodically (weekly/monthly is
plenty) to refresh; the live app just reads the resulting JSON file, no
live API calls needed.
"""

import concurrent.futures
import json
import os
import socket
import time

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

MIN_MARKET_CAP = 1.6e9    # $1.6 billion
MIN_FLOAT_SHARES = 4.0e8  # 400 million shares

STARTUP_COOLDOWN = 600  # seconds -- let any prior rate-limit trip clear first

STAGE1_WORKERS = 5
STAGE1_CHUNK = 400
STAGE1_CHUNK_PAUSE = 15  # seconds between chunks

STAGE2_WORKERS = 3
STAGE2_CHUNK = 40
STAGE2_CHUNK_PAUSE = 12

MAX_RETRIES = 5
BASE_BACKOFF = 30  # seconds, doubles each retry

OUTPUT_PATH = "us_stocks_screened.json"
PROGRESS_PATH = "build_progress.log"
CHECKPOINT_PATH = "build_checkpoint_stage1.json"

socket.setdefaulttimeout(15)


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def get_all_us_tickers():
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
    df = pd.read_csv(url, header=None, names=["Ticker"])
    tickers = df["Ticker"].dropna().astype(str).str.upper().tolist()
    tickers = [t for t in tickers if "." not in t and "-" not in t and len(t) <= 5]
    return sorted(set(tickers))


def with_retry(fn, *args):
    delay = BASE_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args)
        except YFRateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            log(f"    rate-limited, backing off {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2
        except Exception:
            raise


def _fetch_market_cap(ticker):
    # fast_info is a lazy proxy: the real network call happens on .get(),
    # not on attribute access, so both must be inside the retried call.
    return yf.Ticker(ticker).fast_info.get("marketCap")


def cheap_market_cap(ticker):
    try:
        return ticker, with_retry(_fetch_market_cap, ticker)
    except Exception:
        return ticker, None


def _fetch_full_info(ticker):
    return yf.Ticker(ticker).info


def full_info(ticker):
    try:
        info = with_retry(_fetch_full_info, ticker)
        mc = info.get("marketCap", 0) or 0
        float_shares = info.get("floatShares") or info.get("sharesOutstanding", 0) or 0
        name = info.get("shortName") or info.get("longName") or ticker
        return ticker, mc, float_shares, name
    except Exception:
        return ticker, None, None, None


def run_stage1(all_tickers):
    survivors = []
    done = 0
    chunks = list(chunked(all_tickers, STAGE1_CHUNK))
    for ci, chunk in enumerate(chunks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=STAGE1_WORKERS) as ex:
            for ticker, mc in ex.map(cheap_market_cap, chunk):
                done += 1
                if mc and mc >= MIN_MARKET_CAP:
                    survivors.append(ticker)
        log(f"Stage 1: chunk {ci + 1}/{len(chunks)} ({done}/{len(all_tickers)} checked), {len(survivors)} survivors so far")
        if ci < len(chunks) - 1:
            time.sleep(STAGE1_CHUNK_PAUSE)

    log(f"Stage 1 done. {len(survivors)} tickers have market cap >= ${MIN_MARKET_CAP/1e9:.1f}B")
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(survivors, f)
    log(f"Checkpointed stage 1 survivors to {CHECKPOINT_PATH}")
    return survivors


def run_stage2(survivors):
    results = []
    done = 0
    chunks = list(chunked(survivors, STAGE2_CHUNK))
    for ci, chunk in enumerate(chunks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=STAGE2_WORKERS) as ex:
            for ticker, mc, float_shares, name in ex.map(full_info, chunk):
                done += 1
                if mc and float_shares and mc >= MIN_MARKET_CAP and float_shares >= MIN_FLOAT_SHARES:
                    results.append({
                        "ticker": ticker,
                        "name": name,
                        "market_cap": mc,
                        "float_shares": float_shares,
                    })
        log(f"Stage 2: chunk {ci + 1}/{len(chunks)} ({done}/{len(survivors)} checked), {len(results)} passed full screen so far")
        if ci < len(chunks) - 1:
            time.sleep(STAGE2_CHUNK_PAUSE)
    return results


def main():
    open(PROGRESS_PATH, "w").close()
    log(f"Cooling down {STARTUP_COOLDOWN}s before making any requests (prior run tripped rate limiting)...")
    time.sleep(STARTUP_COOLDOWN)

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            survivors = json.load(f)
        log(f"Resuming from checkpoint: {len(survivors)} stage-1 survivors")
    else:
        all_tickers = get_all_us_tickers()
        log(f"Total candidate tickers: {len(all_tickers)}")
        survivors = run_stage1(all_tickers)

    results = run_stage2(survivors)
    results.sort(key=lambda r: r["market_cap"], reverse=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log(f"DONE. {len(results)} tickers passed the full screen. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
