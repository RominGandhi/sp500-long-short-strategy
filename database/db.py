"""
PostgreSQL interface via SQLAlchemy + psycopg2.
Works with Neon (and any standard PostgreSQL).
"""

import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")
CHUNK_SIZE   = 5000


def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


# ── Schema ────────────────────────────────────────────────────────────────────

def create_schema():
    sql = """
    CREATE TABLE IF NOT EXISTS prices (
        date    DATE NOT NULL,
        ticker  TEXT NOT NULL,
        close   NUMERIC,
        PRIMARY KEY (date, ticker)
    );
    CREATE TABLE IF NOT EXISTS index_prices (
        date    DATE NOT NULL PRIMARY KEY,
        open    NUMERIC,
        high    NUMERIC,
        low     NUMERIC,
        close   NUMERIC,
        volume  BIGINT
    );
    CREATE TABLE IF NOT EXISTS returns (
        date        DATE NOT NULL,
        ticker      TEXT NOT NULL,
        log_return  NUMERIC,
        PRIMARY KEY (date, ticker)
    );
    CREATE TABLE IF NOT EXISTS pit_universe (
        date    DATE NOT NULL,
        ticker  TEXT NOT NULL,
        PRIMARY KEY (date, ticker)
    );
    CREATE TABLE IF NOT EXISTS fundamentals (
        ticker              TEXT NOT NULL,
        period_end          DATE NOT NULL,
        available_date      DATE,
        revenue             NUMERIC,
        net_income          NUMERIC,
        total_assets        NUMERIC,
        equity              NUMERIC,
        free_cash_flow      NUMERIC,
        shares_outstanding  NUMERIC,
        PRIMARY KEY (ticker, period_end)
    );
    CREATE INDEX IF NOT EXISTS idx_prices_ticker      ON prices (ticker, date);
    CREATE INDEX IF NOT EXISTS idx_returns_ticker     ON returns (ticker, date);
    CREATE INDEX IF NOT EXISTS idx_pit_date           ON pit_universe (date);
    CREATE INDEX IF NOT EXISTS idx_fundamentals_avail ON fundamentals (ticker, available_date);
    """
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    print("Schema ready.")


# ── Bulk upsert ───────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([float("inf"), float("-inf")], None).where(pd.notnull(df), None)


def upsert_df(df: pd.DataFrame, table: str, chunk_size: int = CHUNK_SIZE):
    """Bulk upsert using PostgreSQL INSERT ... ON CONFLICT DO UPDATE."""
    if df.empty:
        return 0

    df = _clean(df)
    engine = get_engine()
    cols   = list(df.columns)
    pks    = {"prices": ["date","ticker"], "index_prices": ["date"],
              "returns": ["date","ticker"], "pit_universe": ["date","ticker"],
              "fundamentals": ["ticker","period_end"]}
    pk     = pks.get(table, [])
    update = [c for c in cols if c not in pk]

    col_list    = ", ".join(cols)
    placeholder = ", ".join(f":{c}" for c in cols)
    conflict    = ", ".join(pk)
    set_clause  = ", ".join(f"{c} = EXCLUDED.{c}" for c in update)

    sql = text(f"""
        INSERT INTO {table} ({col_list})
        VALUES ({placeholder})
        ON CONFLICT ({conflict}) DO UPDATE SET {set_clause}
    """)

    total = 0
    with engine.connect() as conn:
        for i in range(0, len(df), chunk_size):
            batch = df.iloc[i : i + chunk_size].to_dict(orient="records")
            conn.execute(sql, batch)
            total += len(batch)
        conn.commit()

    return total


# ── Query helpers ─────────────────────────────────────────────────────────────

def query(sql: str, params: dict = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    return query(
        "SELECT date, close FROM prices WHERE ticker=:t AND date BETWEEN :s AND :e ORDER BY date",
        {"t": ticker, "s": start, "e": end}
    )


def get_universe_on(date: str) -> list:
    df = query("SELECT ticker FROM pit_universe WHERE date=:d", {"d": date})
    return df["ticker"].tolist()


def get_fundamentals_as_of(ticker: str, signal_date: str):
    df = query("""
        SELECT * FROM fundamentals
        WHERE ticker=:t AND available_date<=:d
        ORDER BY period_end DESC LIMIT 1
    """, {"t": ticker, "d": signal_date})
    return df.iloc[0] if not df.empty else None


def get_returns_matrix(start: str, end: str) -> pd.DataFrame:
    df = query("""
        SELECT r.date, r.ticker, r.log_return
        FROM returns r
        JOIN pit_universe u ON r.date=u.date AND r.ticker=u.ticker
        WHERE r.date BETWEEN :s AND :e
        ORDER BY r.date
    """, {"s": start, "e": end})
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="log_return")
