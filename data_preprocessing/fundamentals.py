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

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "fundamentals")
REPORTING_LAG_DAYS = 60


# ── SimFin (primary, full history) ───────────────────────────────────────────

def pull_simfin(api_key: str, start: str = "2015-01-01") -> None:
    """
    Pull full historical fundamentals for all US stocks via SimFin.
    Free API key at https://app.simfin.com/api/users (no credit card).
    Downloads bulk CSVs once (~200 MB) and caches them locally.
    """
    import simfin as sf

    sf.set_api_key(api_key)
    sf.set_data_dir(os.path.join(ROOT, "data", "simfin_cache"))

    print("Downloading SimFin bulk data (first run ~200 MB, cached after)...")
    income   = sf.load_income(variant="quarterly",  market="us")
    balance  = sf.load_balance(variant="quarterly", market="us")
    cashflow = sf.load_cashflow(variant="quarterly", market="us")

    print("Processing and splitting by ticker...")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Each df is indexed by (Ticker, Report Date)
    inc = income.reset_index()
    bal = balance.reset_index()
    cf  = cashflow.reset_index()

    # Standardise column names to what we need
    inc = inc.rename(columns={
        "Ticker":           "ticker",
        "Report Date":      "period_end",
        "Revenue":          "revenue",
        "Net Income":       "net_income",
        "Shares (Diluted)": "shares_outstanding",
        "Publish Date":     "publish_date",
    })
    bal = bal.rename(columns={
        "Ticker":        "ticker",
        "Report Date":   "period_end",
        "Total Assets":  "total_assets",
        "Total Equity":  "equity",
    })
    cf = cf.rename(columns={
        "Ticker":                           "ticker",
        "Report Date":                      "period_end",
        "Net Cash from Operating Activities": "cfo",
        "Change in Fixed Assets & Intangibles": "capex",
    })

    # Keep only what we need before merge (avoids column collision)
    inc_cols = ["ticker", "period_end", "publish_date", "revenue", "net_income", "shares_outstanding"]
    bal_cols = ["ticker", "period_end", "total_assets", "equity"]
    cf_cols  = ["ticker", "period_end", "cfo", "capex"]

    inc = inc[[c for c in inc_cols if c in inc.columns]]
    bal = bal[[c for c in bal_cols if c in bal.columns]]
    cf  = cf[[c for c in cf_cols  if c in cf.columns]]

    merged = inc.merge(bal, on=["ticker", "period_end"], how="left")
    merged = merged.merge(cf,  on=["ticker", "period_end"], how="left")

    merged["period_end"]   = pd.to_datetime(merged["period_end"])
    merged["publish_date"] = pd.to_datetime(merged.get("publish_date"), errors="coerce")

    # Use actual publish date if available, else period_end + 60 days
    merged["available_date"] = merged["publish_date"].combine_first(
        merged["period_end"] + pd.Timedelta(days=REPORTING_LAG_DAYS)
    )

    # Free cash flow = operating cash flow - capex
    if "cfo" in merged.columns and "capex" in merged.columns:
        merged["free_cash_flow"] = merged["cfo"] - merged["capex"].abs()

    merged = merged[merged["period_end"] >= pd.Timestamp(start)]
    merged = merged.sort_values(["ticker", "period_end"])

    keep = ["ticker", "period_end", "available_date", "revenue", "net_income",
            "total_assets", "equity", "free_cash_flow", "shares_outstanding"]
    keep = [c for c in keep if c in merged.columns]
    merged = merged[keep]

    saved = 0
    for ticker, grp in merged.groupby("ticker"):
        grp.reset_index(drop=True).to_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), index=False)
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
