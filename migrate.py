"""
One-time migration: load all CSV data into Neon PostgreSQL.
Uses fast bulk COPY via pandas to_sql() — much faster than row-by-row upserts.

Run modes:
  python3 migrate.py          # push current quarter only (default for CI)
  python3 migrate.py --full   # full historical load (run once on setup)
"""

import os
import pandas as pd
from db import get_engine, create_schema

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _wide_to_long(csv_path: str, value_name: str) -> pd.DataFrame:
    """Read a wide prices/returns CSV and melt to long format."""
    df = pd.read_csv(csv_path, skiprows=[1, 2] if _has_multiindex(csv_path) else [],
                     index_col=0, parse_dates=True)
    df.index.name = "date"
    long = (df.reset_index()
              .melt(id_vars="date", var_name="ticker", value_name=value_name)
              .dropna(subset=[value_name]))
    long["date"] = pd.to_datetime(long["date"]).dt.date
    return long


def _has_multiindex(path: str) -> bool:
    with open(path) as f:
        second = f.readlines()[1] if sum(1 for _ in open(path)) > 1 else ""
    return second.startswith("Ticker")


def _bulk_insert(df: pd.DataFrame, table: str, chunk: int = 10000):
    """Fast bulk insert using to_sql COPY-style."""
    engine = get_engine()
    df.to_sql(table, engine, if_exists="append", index=False,
              chunksize=chunk, method="multi")


def migrate_index(quarters: list = None):
    if quarters is None:
        quarters = sorted(d for d in os.listdir(DATA_DIR)
                          if os.path.isdir(os.path.join(DATA_DIR, d)) and d[0].isdigit())
    frames = []
    for q in quarters:
        p = os.path.join(DATA_DIR, q, "index.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p, skiprows=[1, 2], index_col=0, parse_dates=True))

    if not frames:
        return
    df = pd.concat(frames).drop_duplicates()
    df.index.name = "date"
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"adj close": "close", "price": "close"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna(subset=["close"])
    _bulk_insert(df, "index_prices")
    print(f"  index_prices: {len(df):,} rows")


def migrate_prices_and_returns(quarters: list):
    total_p = total_r = 0
    for q in quarters:
        qdir = os.path.join(DATA_DIR, q)

        p = os.path.join(qdir, "prices.csv")
        if os.path.exists(p):
            df = _wide_to_long(p, "close")
            _bulk_insert(df[["date", "ticker", "close"]], "prices")
            total_p += len(df)

        r = os.path.join(qdir, "returns.csv")
        if os.path.exists(r):
            df = _wide_to_long(r, "log_return")
            _bulk_insert(df[["date", "ticker", "log_return"]], "returns")
            total_r += len(df)

        print(f"  {q} done", flush=True)

    print(f"  prices total:  {total_p:,}")
    print(f"  returns total: {total_r:,}")


def migrate_universe():
    path = os.path.join(DATA_DIR, "universe", "pit_membership.csv")
    if not os.path.exists(path):
        print("  No pit_membership.csv — run universe.py first")
        return
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    _bulk_insert(df[["date", "ticker"]], "pit_universe")
    print(f"  pit_universe: {len(df):,} rows")


def migrate_fundamentals():
    fund_dir = os.path.join(DATA_DIR, "fundamentals")
    if not os.path.exists(fund_dir):
        return
    files = [f for f in os.listdir(fund_dir) if f.endswith(".csv")]
    frames = []
    for fname in files:
        ticker = fname.replace(".csv", "")
        df = pd.read_csv(os.path.join(fund_dir, fname),
                         parse_dates=["period_end", "available_date"])
        if "ticker" not in df.columns:
            df.insert(0, "ticker", ticker)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["period_end"]     = pd.to_datetime(combined["period_end"]).dt.date
    combined["available_date"] = pd.to_datetime(combined["available_date"]).dt.date
    keep = ["ticker", "period_end", "available_date", "revenue", "net_income",
            "total_assets", "equity", "free_cash_flow", "shares_outstanding"]
    combined = combined[[c for c in keep if c in combined.columns]]
    _bulk_insert(combined, "fundamentals")
    print(f"  fundamentals: {len(combined):,} rows")


def run_full():
    """Truncate all tables and reload everything from scratch."""
    engine = get_engine()
    with engine.connect() as conn:
        from sqlalchemy import text
        for t in ["prices", "returns", "index_prices", "pit_universe", "fundamentals"]:
            conn.execute(text(f"TRUNCATE TABLE {t}"))
        conn.commit()
    print("Tables cleared.")

    quarters = sorted(d for d in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, d)) and d[0].isdigit())

    print("\n── Index prices ──")
    migrate_index()

    print("\n── Prices + Returns ──")
    migrate_prices_and_returns(quarters)

    print("\n── Universe ──")
    migrate_universe()

    print("\n── Fundamentals ──")
    migrate_fundamentals()

    print("\nMigration complete.")


def push_current_quarter():
    """Push only the current quarter — used by GitHub Actions."""
    from datetime import date
    today = pd.Timestamp(date.today())
    q = today.to_period("Q")
    label = f"{q.year}_Q{q.quarter}"
    print(f"Pushing {label} to Neon...")

    engine = get_engine()
    with engine.connect() as conn:
        from sqlalchemy import text
        # Remove existing rows for this quarter's date range
        start = str(q.start_time.date())
        end   = str(q.end_time.date())
        conn.execute(text(f"DELETE FROM prices       WHERE date BETWEEN '{start}' AND '{end}'"))
        conn.execute(text(f"DELETE FROM returns      WHERE date BETWEEN '{start}' AND '{end}'"))
        conn.execute(text(f"DELETE FROM index_prices WHERE date BETWEEN '{start}' AND '{end}'"))
        conn.execute(text(f"DELETE FROM pit_universe WHERE date BETWEEN '{start}' AND '{end}'"))
        conn.execute(text(f"DELETE FROM fundamentals WHERE available_date BETWEEN '{start}' AND '{end}'"))
        conn.commit()

    migrate_prices_and_returns([label])
    migrate_index(quarters=[label])

    # Universe: only insert rows for this quarter's date range
    path = os.path.join(DATA_DIR, "universe", "pit_membership.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= start) & (df["date"] <= end)
        df = df[mask].copy()
        df["date"] = df["date"].dt.date
        _bulk_insert(df[["date", "ticker"]], "pit_universe")
        print(f"  pit_universe: {len(df):,} rows")

    # Fundamentals: only insert rows with available_date in this quarter
    fund_dir = os.path.join(DATA_DIR, "fundamentals")
    if os.path.exists(fund_dir):
        files = [f for f in os.listdir(fund_dir) if f.endswith(".csv")]
        frames = []
        for fname in files:
            ticker = fname.replace(".csv", "")
            df = pd.read_csv(os.path.join(fund_dir, fname),
                             parse_dates=["period_end", "available_date"])
            if "ticker" not in df.columns:
                df.insert(0, "ticker", ticker)
            mask = (df["available_date"] >= start) & (df["available_date"] <= end)
            if mask.any():
                frames.append(df[mask])
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined["period_end"]     = pd.to_datetime(combined["period_end"]).dt.date
            combined["available_date"] = pd.to_datetime(combined["available_date"]).dt.date
            keep = ["ticker", "period_end", "available_date", "revenue", "net_income",
                    "total_assets", "equity", "free_cash_flow", "shares_outstanding"]
            combined = combined[[c for c in keep if c in combined.columns]]
            _bulk_insert(combined, "fundamentals")
            print(f"  fundamentals: {len(combined):,} rows")

    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Full reload from scratch")
    args = parser.parse_args()
    if args.full:
        run_full()
    else:
        push_current_quarter()
