"""
Step 1.3 — Quarterly fundamental data with reporting lag

Sources (in order of preference):
  1. SimFin (free API key at simfin.com — no credit card) — full history to 2010+
  2. yfinance fallback — last ~5 quarters only, good for recent data

Reporting lag: A Q1 report (period ending March 31) isn't public until ~May 30.
Every row is stamped with available_date = period_end + 60 days.
Your strategy must only use rows where available_date <= signal_date.

Output: data/fundamentals/<TICKER>.csv
  columns: period_end, available_date, revenue, net_income, total_assets,
           equity, free_cash_flow, shares_outstanding, eps
"""

import os
import time
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "fundamentals")
REPORTING_LAG_DAYS = 60


# ── SimFin (primary, full history) ───────────────────────────────────────────

def pull_simfin(api_key: str, start: str = "2015-01-01") -> None:
    """
    Pull full historical fundamentals for all US stocks via SimFin.
    Free API key at https://app.simfin.com/api/users (no credit card).
    Downloads bulk CSVs once and caches them locally.
    """
    try:
        import simfin as sf
        from simfin.names import (REVENUE, NET_INCOME, TOTAL_ASSETS,
                                  SHARES_DILUTED, FREE_CASH_FLOW)
    except ImportError:
        print("Install simfin: pip install simfin")
        return

    sf.set_api_key(api_key)
    sf.set_data_dir(os.path.join(os.path.dirname(__file__), "data", "simfin_cache"))

    print("Downloading SimFin bulk data (first run downloads ~200 MB, cached after)...")

    income  = sf.load_income(variant="quarterly",  market="us")
    balance = sf.load_balance(variant="quarterly", market="us")
    cashflow = sf.load_cashflow(variant="quarterly", market="us")

    print("Processing SimFin data...")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Merge on (Ticker, Report Date)
    merged = income.join(balance,  how="left", rsuffix="_bal")
    merged = merged.join(cashflow, how="left", rsuffix="_cf")
    merged = merged.reset_index()

    # Normalise column names
    col_map = {
        "Ticker":          "ticker",
        "Report Date":     "period_end",
        "Revenue":         "revenue",
        "Net Income":      "net_income",
        "Total Assets":    "total_assets",
        "Shares (Diluted)":"shares_outstanding",
        "Free Cash Flow":  "free_cash_flow",
    }
    merged = merged.rename(columns={k: v for k, v in col_map.items() if k in merged.columns})

    if "period_end" not in merged.columns:
        print("  Warning: could not find 'Report Date' column in SimFin data")
        return

    merged["period_end"]    = pd.to_datetime(merged["period_end"])
    merged["available_date"] = merged["period_end"] + pd.Timedelta(days=REPORTING_LAG_DAYS)

    # Filter date range
    merged = merged[merged["period_end"] >= pd.Timestamp(start)]

    keep = ["ticker", "period_end", "available_date", "revenue", "net_income",
            "total_assets", "shares_outstanding", "free_cash_flow"]
    keep = [c for c in keep if c in merged.columns]
    merged = merged[keep].sort_values(["ticker", "period_end"])

    saved = 0
    for ticker, grp in merged.groupby("ticker"):
        grp = grp.reset_index(drop=True)
        grp.to_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), index=False)
        saved += 1

    print(f"SimFin: saved {saved} tickers to {DATA_DIR}/")


# ── yfinance fallback (last ~5 quarters only) ────────────────────────────────

def pull_ticker_yfinance(ticker: str):
    try:
        t = yf.Ticker(ticker)
        income  = t.quarterly_income_stmt
        balance = t.quarterly_balance_sheet
        cash    = t.quarterly_cash_flow

        if income is None or income.empty:
            return None

        def safe_row(df, *keys):
            for k in keys:
                if df is not None and k in df.index:
                    return df.loc[k]
            return pd.Series(dtype=float)

        revenue      = safe_row(income,  "Total Revenue")
        net_income   = safe_row(income,  "Net Income")
        shares       = safe_row(income,  "Diluted Average Shares", "Basic Average Shares")
        eps          = safe_row(income,  "Diluted EPS", "Basic EPS")
        total_assets = safe_row(balance, "Total Assets")
        equity       = safe_row(balance, "Stockholders Equity", "Common Stock Equity")
        fcf          = safe_row(cash,    "Free Cash Flow")

        rows = []
        for p in income.columns:
            rows.append({
                "period_end":         pd.Timestamp(p).date(),
                "available_date":     (pd.Timestamp(p) + pd.Timedelta(days=REPORTING_LAG_DAYS)).date(),
                "revenue":            revenue.get(p),
                "net_income":         net_income.get(p),
                "total_assets":       total_assets.get(p),
                "equity":             equity.get(p),
                "free_cash_flow":     fcf.get(p),
                "shares_outstanding": shares.get(p),
                "eps":                eps.get(p),
            })

        df = pd.DataFrame(rows).sort_values("period_end").reset_index(drop=True)
        return df

    except Exception:
        return None


def pull_yfinance_fallback(tickers: list[str], delay: float = 0.3) -> None:
    """
    Pull recent fundamentals via yfinance for tickers not yet in data/fundamentals/.
    NOTE: yfinance only returns the last ~5 quarters. Use SimFin for full history.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    ok = skipped = 0

    for i, ticker in enumerate(tickers):
        out_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if os.path.exists(out_path):
            ok += 1
            continue

        df = pull_ticker_yfinance(ticker)
        if df is not None and not df.empty:
            df.to_csv(out_path, index=False)
            ok += 1
        else:
            skipped += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tickers)} — saved {ok}, skipped {skipped}")
        time.sleep(delay)

    print(f"yfinance fallback done. {ok} saved, {skipped} skipped.")


# ── Query interface ───────────────────────────────────────────────────────────

def load_fundamentals(ticker: str):
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["period_end", "available_date"])


def get_fundamentals_as_of(ticker: str, signal_date):
    """
    Return the most recent fundamental row visible on signal_date
    (respects the 60-day reporting lag — no look-ahead bias).
    """
    df = load_fundamentals(ticker)
    if df is None:
        return None
    signal_date = pd.Timestamp(signal_date)
    available = df[df["available_date"] <= signal_date]
    if available.empty:
        return None
    return available.iloc[-1]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simfin-key", help="SimFin API key (free at simfin.com)")
    parser.add_argument("--yfinance",   action="store_true", help="Use yfinance fallback instead")
    parser.add_argument("--tickers",    nargs="*", help="Specific tickers (default: all ever-in-index)")
    args = parser.parse_args()

    ever_path = os.path.join(os.path.dirname(__file__), "data", "universe", "all_tickers_ever.csv")
    if args.tickers:
        tickers = args.tickers
    elif os.path.exists(ever_path):
        tickers = pd.read_csv(ever_path)["ticker"].tolist()
    else:
        print("Run universe.py first to generate the ticker list.")
        exit(1)

    if args.simfin_key:
        pull_simfin(api_key=args.simfin_key)
    elif args.yfinance:
        print(f"WARNING: yfinance only returns ~5 quarters. Use --simfin-key for full history.")
        print(f"Pulling {len(tickers)} tickers via yfinance...")
        pull_yfinance_fallback(tickers)
    else:
        print("Specify --simfin-key YOUR_KEY  or  --yfinance")
        print("Free SimFin API key: https://app.simfin.com/api/users")
