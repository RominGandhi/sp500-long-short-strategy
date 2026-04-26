"""
Supabase interface for the S&P 500 long/short pipeline.
Uses raw HTTP requests — no supabase-py client needed, works on all Python versions.
"""

import os
import math
import time
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://fvoanmhluogdzvbxcsat.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
CHUNK_SIZE   = 500


def _headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }


def _rest(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


# ── Bulk upsert ───────────────────────────────────────────────────────────────

def upsert_df(df: pd.DataFrame, table: str, chunk_size: int = CHUNK_SIZE):
    """Upsert a DataFrame into a Supabase table using direct HTTP with retries."""
    if df.empty:
        return 0

    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.where(pd.notnull(df), None)

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
    headers = {**_headers(), "Prefer": "resolution=merge-duplicates"}
    url = _rest(table)
    total = 0

    for i in range(0, len(records), chunk_size):
        batch = records[i : i + chunk_size]
        for attempt in range(5):
            r = requests.post(url, json=batch, headers=headers)
            if r.status_code in (200, 201):
                break
            if attempt == 4:
                raise RuntimeError(f"Supabase upsert failed [{r.status_code}]: {r.text[:300]}")
            time.sleep(2 ** attempt)
        total += len(batch)

    return total


# ── Query helpers ─────────────────────────────────────────────────────────────

def _select(table: str, params: dict) -> list:
    r = requests.get(_rest(table), headers=_headers(), params=params)
    r.raise_for_status()
    return r.json()


def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = _select("prices", {
        "select": "date,close",
        "ticker": f"eq.{ticker}",
        "date":   f"gte.{start}",
        "order":  "date",
    })
    df = pd.DataFrame(data)
    df = df[df["date"] <= end] if not df.empty else df
    return df


def get_universe_on(date: str) -> list:
    data = _select("pit_universe", {"select": "ticker", "date": f"eq.{date}"})
    return [r["ticker"] for r in data]


def get_fundamentals_as_of(ticker: str, signal_date: str):
    data = _select("fundamentals", {
        "select":         "*",
        "ticker":         f"eq.{ticker}",
        "available_date": f"lte.{signal_date}",
        "order":          "period_end.desc",
        "limit":          "1",
    })
    return pd.Series(data[0]) if data else None


def get_returns_matrix(start: str, end: str, tickers: list = None) -> pd.DataFrame:
    if tickers is None:
        data = _select("pit_universe", {
            "select": "ticker",
            "date":   f"gte.{start}",
        })
        tickers = list({r["ticker"] for r in data})

    all_rows = []
    for i in range(0, len(tickers), 50):
        batch = tickers[i : i + 50]
        data = _select("returns", {
            "select": "date,ticker,log_return",
            "ticker": f"in.({','.join(batch)})",
            "date":   f"gte.{start}",
            "order":  "date",
        })
        all_rows.extend(data)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df[df["date"] <= end]
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="log_return")


def ping() -> bool:
    """Return True if Supabase is reachable and schema is ready."""
    try:
        r = requests.get(
            _rest("prices"),
            headers=_headers(),
            params={"select": "date", "limit": "1"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False
