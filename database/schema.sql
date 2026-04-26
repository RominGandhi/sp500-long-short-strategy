-- S&P 500 Long/Short Strategy — Supabase Schema
-- Run this once in: Supabase Dashboard → SQL Editor → New Query

CREATE TABLE IF NOT EXISTS prices (
    date    DATE    NOT NULL,
    ticker  TEXT    NOT NULL,
    close   NUMERIC,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS index_prices (
    date    DATE    NOT NULL PRIMARY KEY,
    open    NUMERIC,
    high    NUMERIC,
    low     NUMERIC,
    close   NUMERIC,
    volume  BIGINT
);

CREATE TABLE IF NOT EXISTS returns (
    date        DATE    NOT NULL,
    ticker      TEXT    NOT NULL,
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

-- Indexes for fast signal queries
CREATE INDEX IF NOT EXISTS idx_prices_ticker      ON prices (ticker, date);
CREATE INDEX IF NOT EXISTS idx_returns_ticker     ON returns (ticker, date);
CREATE INDEX IF NOT EXISTS idx_pit_date           ON pit_universe (date);
CREATE INDEX IF NOT EXISTS idx_fundamentals_avail ON fundamentals (ticker, available_date);
