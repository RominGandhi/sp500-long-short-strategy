"""
Supabase interface for the S&P 500 long/short pipeline.
Uses the supabase-py REST client — no direct DB password needed.

Tables: prices, index_prices, returns, pit_universe, fundamentals
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://fvoanmhluogdzvbxcsat.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
CHUNK_SIZE   = 1000  # rows per REST API call


def client():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Bulk upsert ───────────────────────────────────────────────────────────────

def upsert_df(df: pd.DataFrame, table: str, chunk_size: int = CHUNK_SIZE):
    """Upsert a DataFrame into a Supabase table in batches."""
    if df.empty:
        return 0

    import math, numpy as np
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.where(pd.notnull(df), None)

    # Convert every value to a JSON-safe Python native type
    def _safe(v):
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return None if math.isnan(float(v)) else float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    records = [{k: _safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]
    sb = client()
    total = 0

    for i in range(0, len(records), chunk_size):
        batch = records[i : i + chunk_size]
        for attempt in range(4):
            try:
                sb.table(table).upsert(batch).execute()
                break
            except Exception as e:
                if attempt == 3:
                    raise
                import time
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
        total += len(batch)

    return total


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    sb = client()
    res = (sb.table("prices")
             .select("date, close")
             .eq("ticker", ticker)
             .gte("date", start)
             .lte("date", end)
             .order("date")
             .execute())
    return pd.DataFrame(res.data)


def get_universe_on(date: str) -> list:
    sb = client()
    res = sb.table("pit_universe").select("ticker").eq("date", date).execute()
    return [r["ticker"] for r in res.data]


def get_fundamentals_as_of(ticker: str, signal_date: str):
    """Most recent fundamental row available on signal_date (no look-ahead)."""
    sb = client()
    res = (sb.table("fundamentals")
             .select("*")
             .eq("ticker", ticker)
             .lte("available_date", signal_date)
             .order("period_end", desc=True)
             .limit(1)
             .execute())
    if not res.data:
        return None
    return pd.Series(res.data[0])


def get_returns_matrix(start: str, end: str, tickers: list = None) -> pd.DataFrame:
    """
    Returns a (date × ticker) pivot of log returns for tickers
    that were in the index on each date.
    Uses chunked fetches to stay within API limits.
    """
    sb = client()

    # Get universe tickers for the date range if not specified
    if tickers is None:
        res = (sb.table("pit_universe")
                 .select("ticker")
                 .gte("date", start)
                 .lte("date", end)
                 .execute())
        tickers = list({r["ticker"] for r in res.data})

    all_rows = []
    for i in range(0, len(tickers), 50):
        batch = tickers[i : i + 50]
        res = (sb.table("returns")
                 .select("date, ticker, log_return")
                 .in_("ticker", batch)
                 .gte("date", start)
                 .lte("date", end)
                 .execute())
        all_rows.extend(res.data)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="log_return")


def query_sql(sql: str) -> pd.DataFrame:
    """Run raw SQL via a Supabase RPC function named 'run_sql' if configured."""
    sb = client()
    res = sb.rpc("run_sql", {"query": sql}).execute()
    return pd.DataFrame(res.data)
