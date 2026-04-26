"""
S&P 500 Data Pipeline
Pulls constituent and index data from Yahoo Finance (free, no API key required).
Saves quarterly slices to ./data/YYYY_QN/ subfolders.

Run modes:
  python3 data_pipeline.py --full          # pull everything from --start
  python3 data_pipeline.py --update        # only pull missing/incomplete quarters (default)
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
LOG_DIR  = os.path.join(ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "update_log.csv")


def quarter_label(period: pd.Period) -> str:
    return f"{period.year}_Q{period.quarter}"


def existing_quarters() -> set[str]:
    if not os.path.isdir(DATA_DIR):
        return set()
    return {
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d[0].isdigit()
    }


def current_quarter() -> str:
    today = pd.Timestamp(date.today())
    return quarter_label(today.to_period("Q"))


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    import requests
    from io import StringIO

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    table = pd.read_html(StringIO(resp.text))[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    meta = table[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    meta["Symbol"] = tickers
    # Stamp with today so we know which constituents were active each quarter
    meta["as_of"] = str(date.today())
    meta.to_csv(os.path.join(DATA_DIR, "sp500_constituents.csv"), index=False)
    print(f"Fetched {len(tickers)} current S&P 500 constituents")
    return tickers


def download_index(start: str) -> pd.DataFrame:
    print(f"Downloading ^GSPC from {start}...")
    df = yf.download("^GSPC", start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("No index data returned — check your internet connection")
    print(f"  {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def download_constituents(tickers: list[str], start: str, batch_size: int = 50) -> pd.DataFrame:
    print(f"Downloading constituent prices from {start} in batches of {batch_size}...")
    all_closes = []
    n_batches = -(-len(tickers) // batch_size)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{n_batches}: {len(batch)} tickers")
        try:
            raw = yf.download(batch, start=start, auto_adjust=True, progress=False, threads=True)
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            else:
                closes = raw[["Close"]].rename(columns={"Close": batch[0]})
            all_closes.append(closes)
        except Exception as e:
            print(f"  Warning: batch failed ({e}), skipping")
        time.sleep(1)

    prices = pd.concat(all_closes, axis=1)
    prices.sort_index(inplace=True)
    return prices


def save_by_quarter(index_df: pd.DataFrame, prices: pd.DataFrame, overwrite_current: bool = True):
    """Split data by quarter and write each to its subfolder."""
    returns = np.log(prices / prices.shift(1)).dropna(how="all")

    price_quarters = prices.index.to_period("Q")
    return_quarters = returns.index.to_period("Q")
    index_quarters = index_df.index.to_period("Q")
    unique_quarters = sorted(set(price_quarters))

    cq = current_quarter()
    saved = []

    for q in unique_quarters:
        label = quarter_label(q)

        # Skip already-complete quarters unless this is the current (partial) one
        qdir = os.path.join(DATA_DIR, label)
        if os.path.exists(qdir) and label != cq and not overwrite_current:
            continue

        os.makedirs(qdir, exist_ok=True)

        prices.loc[price_quarters == q].to_csv(os.path.join(qdir, "prices.csv"))
        returns.loc[return_quarters == q].to_csv(os.path.join(qdir, "returns.csv"))
        index_df.loc[index_quarters == q].to_csv(os.path.join(qdir, "index.csv"))

        n_days = (price_quarters == q).sum()
        status = "(current — partial)" if label == cq else ""
        print(f"  Saved {label}: {n_days} trading days {status}")
        saved.append(label)

    return saved


def log_run(quarters_saved: list[str]):
    """Append a record to update_log.csv so we can audit what ran when."""
    os.makedirs(LOG_DIR, exist_ok=True)
    entry = pd.DataFrame([{
        "run_at": pd.Timestamp.now().isoformat(),
        "quarters_saved": ",".join(quarters_saved) if quarters_saved else "none",
    }])
    if os.path.exists(LOG_FILE):
        entry.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        entry.to_csv(LOG_FILE, index=False)


def print_summary():
    quarters = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d[0].isdigit()
    )
    print(f"\n{'Quarter':<12} {'Prices':>12} {'Returns':>12} {'Index':>10}")
    print("-" * 50)
    for q in quarters:
        qdir = os.path.join(DATA_DIR, q)
        def kb(fname):
            p = os.path.join(qdir, fname)
            return f"{os.path.getsize(p)/1024:.1f} KB" if os.path.exists(p) else "-"
        marker = " <-- current" if q == current_quarter() else ""
        print(f"  {q:<12} {kb('prices.csv'):>12} {kb('returns.csv'):>12} {kb('index.csv'):>10}{marker}")


def run_update():
    """
    Incremental mode: re-fetch constituents, then pull only quarters
    that don't exist yet plus the current (partial) quarter.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    have = existing_quarters()
    cq = current_quarter()
    print(f"Existing quarters: {len(have)}  |  Current quarter: {cq}")

    # Always refresh constituents — membership changes quarterly
    tickers = get_sp500_tickers()

    # Find the start date: day after the last complete quarter we have,
    # or the start of the current quarter if we only have current.
    complete = sorted(q for q in have if q != cq)
    if complete:
        last = pd.Period(complete[-1].replace("_Q", "Q"), freq="Q")
        start = str((last + 1).start_time.date())
    else:
        start = str(pd.Period(cq.replace("_Q", "Q"), freq="Q").start_time.date())

    print(f"Pulling data from {start} onward...")
    index_df = download_index(start=start)
    prices = download_constituents(tickers, start=start)

    saved = save_by_quarter(index_df, prices, overwrite_current=True)
    log_run(saved)
    print_summary()
    print(f"\nUpdate complete. {len(saved)} quarter(s) written.")


def run_full(start: str = "2015-01-01"):
    """Full backfill from scratch."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tickers = get_sp500_tickers()
    index_df = download_index(start=start)
    prices = download_constituents(tickers, start=start)
    saved = save_by_quarter(index_df, prices, overwrite_current=True)
    log_run(saved)
    print_summary()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="S&P 500 quarterly data pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--update", action="store_true", default=True,
                       help="Incremental: pull only new/current quarter (default)")
    group.add_argument("--full", action="store_true",
                       help="Full backfill from --start date")
    parser.add_argument("--start", default="2015-01-01", help="Start date for --full mode")
    args = parser.parse_args()

    if args.full:
        run_full(start=args.start)
    else:
        run_update()
