"""
One-time migration: load all CSV data into Supabase.

Prerequisites:
  1. Run schema.sql in the Supabase SQL Editor first
  2. Then: python3 migrate.py
"""

import os
import pandas as pd
from db import upsert_df

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def migrate_prices_and_returns():
    quarters = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d[0].isdigit()
    )
    print(f"Migrating {len(quarters)} quarters of prices + returns...")

    for q in quarters:
        qdir = os.path.join(DATA_DIR, q)

        p_path = os.path.join(qdir, "prices.csv")
        if os.path.exists(p_path):
            df = pd.read_csv(p_path, index_col=0, parse_dates=True)
            df.index.name = "date"
            long = (df.reset_index()
                      .melt(id_vars="date", var_name="ticker", value_name="close")
                      .dropna(subset=["close"]))
            long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
            n = upsert_df(long[["date", "ticker", "close"]], "prices")

        r_path = os.path.join(qdir, "returns.csv")
        if os.path.exists(r_path):
            df = pd.read_csv(r_path, index_col=0, parse_dates=True)
            df.index.name = "date"
            long = (df.reset_index()
                      .melt(id_vars="date", var_name="ticker", value_name="log_return")
                      .dropna(subset=["log_return"]))
            long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
            upsert_df(long[["date", "ticker", "log_return"]], "returns")

        print(f"  {q} done")


def migrate_index():
    quarters = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d[0].isdigit()
    )
    frames = []
    for q in quarters:
        ip = os.path.join(DATA_DIR, q, "index.csv")
        if os.path.exists(ip):
            frames.append(pd.read_csv(ip, skiprows=[1, 2], index_col=0, parse_dates=True))

    if not frames:
        return

    df = pd.concat(frames).drop_duplicates()
    df.index.name = "date"
    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df.rename(columns={"adj close": "close", "price": "close"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna(subset=["close"])
    upsert_df(df, "index_prices")
    print(f"Index: {len(df)} rows")


def migrate_universe():
    path = os.path.join(DATA_DIR, "universe", "pit_membership.csv")
    if not os.path.exists(path):
        print("No pit_membership.csv — run universe.py first")
        return
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    upsert_df(df[["date", "ticker"]], "pit_universe")
    print(f"Universe: {len(df):,} rows")


def migrate_fundamentals():
    fund_dir = os.path.join(DATA_DIR, "fundamentals")
    if not os.path.exists(fund_dir):
        print("No fundamentals dir — run fundamentals.py first")
        return

    files = [f for f in os.listdir(fund_dir) if f.endswith(".csv")]
    print(f"Migrating fundamentals: {len(files)} tickers...")

    frames = []
    for fname in files:
        ticker = fname.replace(".csv", "")
        df = pd.read_csv(os.path.join(fund_dir, fname),
                         parse_dates=["period_end", "available_date"])
        if "ticker" not in df.columns:
            df.insert(0, "ticker", ticker)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["period_end"]     = pd.to_datetime(combined["period_end"]).dt.strftime("%Y-%m-%d")
    combined["available_date"] = pd.to_datetime(combined["available_date"]).dt.strftime("%Y-%m-%d")
    keep = ["ticker", "period_end", "available_date", "revenue", "net_income",
            "total_assets", "equity", "free_cash_flow", "shares_outstanding"]
    combined = combined[[c for c in keep if c in combined.columns]]
    upsert_df(combined, "fundamentals")
    print(f"Fundamentals: {len(combined):,} rows")


def run():
    print("── Index prices ──")
    migrate_index()

    print("\n── Prices + Returns ──")
    migrate_prices_and_returns()

    print("\n── Point-in-time universe ──")
    migrate_universe()

    print("\n── Fundamentals ──")
    migrate_fundamentals()

    print("\nMigration complete.")


def _wait_for_supabase(max_wait: int = 30):
    """Wait until Supabase is reachable."""
    import time
    from db import ping
    for _ in range(max_wait // 5):
        if ping():
            return
        time.sleep(5)
    raise RuntimeError("Supabase not reachable after waiting")


def push_current_quarter():
    """
    Lightweight version for GitHub Actions: only push the current quarter's
    prices/returns + refreshed universe + fundamentals. Skips all historical data.
    """
    from datetime import date
    import pandas as pd

    today = pd.Timestamp(date.today())
    q = today.to_period("Q")
    label = f"{q.year}_Q{q.quarter}"
    qdir = os.path.join(DATA_DIR, label)

    print(f"Pushing current quarter: {label}")

    # Prices + returns for current quarter only
    p_path = os.path.join(qdir, "prices.csv")
    if os.path.exists(p_path):
        df = pd.read_csv(p_path, index_col=0, parse_dates=True)
        df.index.name = "date"
        long = (df.reset_index()
                  .melt(id_vars="date", var_name="ticker", value_name="close")
                  .dropna(subset=["close"]))
        long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
        upsert_df(long[["date", "ticker", "close"]], "prices")
        print(f"  Prices: {len(long):,} rows")

    r_path = os.path.join(qdir, "returns.csv")
    if os.path.exists(r_path):
        df = pd.read_csv(r_path, index_col=0, parse_dates=True)
        df.index.name = "date"
        long = (df.reset_index()
                  .melt(id_vars="date", var_name="ticker", value_name="log_return")
                  .dropna(subset=["log_return"]))
        long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
        upsert_df(long[["date", "ticker", "log_return"]], "returns")
        print(f"  Returns: {len(long):,} rows")

    # Index prices for current quarter
    ip = os.path.join(qdir, "index.csv")
    if os.path.exists(ip):
        df = pd.read_csv(ip, skiprows=[1, 2], index_col=0, parse_dates=True)
        df.index.name = "date"
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"adj close": "close", "price": "close"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
        upsert_df(df[keep].dropna(subset=["close"]), "index_prices")
        print(f"  Index: {len(df):,} rows")

    # Full universe refresh (small file, always re-push)
    migrate_universe()

    # Fundamentals refresh
    migrate_fundamentals()

    print(f"Done pushing {label} to Supabase.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Migrate all historical data")
    parser.add_argument("--quarter", action="store_true", help="Push current quarter only (default for CI)")
    args = parser.parse_args()

    if args.full:
        run()
    else:
        push_current_quarter()
