"""
SEC EDGAR Fundamentals Puller
Pulls directly from data.sec.gov — free, official, 100% S&P 500 coverage.

Uses actual SEC filing dates as available_date for exact point-in-time accuracy.

For flow items (net_income, operating_cf, capex):
  - Standalone quarters only: uses the XBRL 'start' date to filter to periods
    of 60-110 days, avoiding YTD cumulative entries that share the same period_end
    and fp as the standalone quarter (common in Apple, Microsoft, etc.)
  - Q4 derived as: FY annual total (10-K) minus 9-month YTD (Q3 cumulative)
    so TTM can cover all four quarters, not just Q1-Q3

Output: data/sec_fundamentals/<TICKER>.csv per company
  columns: ticker, period_end, available_date,
           net_income, total_assets, equity,
           operating_cf, capex, free_cash_flow
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_DIR  = os.path.join(ROOT, "data", "sec_fundamentals")
HEADERS  = {"User-Agent": "sp500-research romingandhi78@gmail.com"}
BASE_URL = "https://data.sec.gov"

CONCEPTS = {
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "total_assets": [
        "Assets",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityAttributableToParent",
        "CommonStockholdersEquity",
    ],
    "operating_cf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquirePropertyPlantAndEquipmentProductiveAssets",
    ],
}

FLOW_ITEMS    = {"net_income", "operating_cf", "capex"}
BALANCE_ITEMS = {"total_assets", "equity"}


# ── CIK mapping ───────────────────────────────────────────────────────────────

def get_cik_map() -> dict:
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return {
        v["ticker"].upper(): str(v["cik_str"]).zfill(10)
        for v in r.json().values()
    }


def fetch_company_facts(cik: str) -> dict:
    url = f"{BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()


# ── Raw concept extractor ─────────────────────────────────────────────────────

def extract_concept(facts: dict, concept_names: list, label: str) -> pd.DataFrame:
    """
    Returns all 10-Q / 10-K entries for the first matching concept as a DataFrame.
    Columns: period_end, available_date, start (if available), form, fp, <label>
    No duration or form filtering — callers decide what to keep.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for name in concept_names:
        if name not in us_gaap:
            continue
        units = us_gaap[name].get("units", {})
        if "USD" not in units and "shares" not in units:
            continue

        unit_key = "USD" if "USD" in units else "shares"
        df = pd.DataFrame(units[unit_key])

        if df.empty or "form" not in df.columns:
            continue

        df = df[df["form"].isin({"10-Q", "10-K", "10-K/A"})].copy()
        df = df[df["filed"].notna()].copy()
        if df.empty:
            continue

        keep = [c for c in ["start", "end", "filed", "val", "form", "fp"] if c in df.columns]
        df = df[keep].rename(columns={"end": "period_end", "filed": "available_date", "val": label})
        df["period_end"]     = pd.to_datetime(df["period_end"])
        df["available_date"] = pd.to_datetime(df["available_date"])
        if "start" in df.columns:
            df["start"] = pd.to_datetime(df["start"])

        return df.reset_index(drop=True)

    return pd.DataFrame()


# ── Flow-item helpers ─────────────────────────────────────────────────────────

def _standalone_quarters(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Filter a raw concept DataFrame to standalone quarterly periods.
    Uses the XBRL 'start' date: a standalone quarter is 60-110 days long.
    Falls back to fp filter if start is unavailable.
    Deduplicates on period_end keeping the first (original) filing.
    """
    if "start" in df.columns and df["start"].notna().any():
        duration = (df["period_end"] - df["start"]).dt.days
        df = df[duration.between(60, 110)].copy()
    else:
        if "fp" in df.columns:
            df = df[df["fp"].isin({"Q1", "Q2", "Q3", "Q4"})].copy()

    # Accept 10-Q, and 10-K where the company tags a standalone Q4 (fp="Q4")
    mask = (df["form"] == "10-Q")
    if "fp" in df.columns:
        mask = mask | (df["form"].isin({"10-K", "10-K/A"}) & (df["fp"] == "Q4"))
    df = df[mask].copy()

    return (df.sort_values("available_date")
              .drop_duplicates("period_end", keep="first")
              .reset_index(drop=True))


def _derive_q4(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Derive Q4 standalone = FY annual total − 9-month Q3 YTD cumulative.
    Both come from the same 10-K / 10-Q filings respectively.
    Returns one row per fiscal year where Q4 can be derived.
    available_date is the 10-K filing date (when Q4 becomes public).
    """
    if "fp" not in df.columns or "start" not in df.columns:
        return pd.DataFrame()

    # Annual totals from 10-K
    fy = df[
        df["fp"].isin({"FY", "CY"}) &
        df["form"].isin({"10-K", "10-K/A"})
    ].copy()
    if fy.empty:
        return pd.DataFrame()

    # 9-month YTD: fp="Q3", duration 250-290 days
    duration = (df["period_end"] - df["start"]).dt.days
    ytd9 = df[
        df["fp"].isin({"Q3", "Q9M"}) &
        duration.between(250, 290)
    ].copy()
    if ytd9.empty:
        return pd.DataFrame()

    results = []
    for _, fy_row in fy.iterrows():
        fy_start  = fy_row["start"]
        fy_end    = fy_row["period_end"]
        fy_filed  = fy_row["available_date"]

        # 9-month YTD with the same fiscal year start, ending before FY end
        match = ytd9[
            (ytd9["start"] == fy_start) &
            (ytd9["period_end"] < fy_end) &
            (ytd9["period_end"] > fy_end - pd.Timedelta(days=120))
        ]
        if match.empty:
            continue

        ytd_val = match.iloc[0][label]
        q4_val  = fy_row[label] - ytd_val

        results.append({
            "period_end":     fy_end,
            "available_date": fy_filed,
            label:            q4_val,
        })

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out["period_end"]     = pd.to_datetime(out["period_end"])
    out["available_date"] = pd.to_datetime(out["available_date"])
    return out


# ── Per-ticker pull ───────────────────────────────────────────────────────────

def pull_ticker(ticker: str, cik: str):
    """Pull and parse all fundamentals for one ticker from SEC EDGAR."""
    facts = fetch_company_facts(cik)
    if not facts:
        return None

    frames = {}
    for label, concepts in CONCEPTS.items():
        raw = extract_concept(facts, concepts, label)
        if raw.empty:
            continue

        if label in FLOW_ITEMS:
            standalone = _standalone_quarters(raw, label)
            q4         = _derive_q4(raw, label)

            if not q4.empty:
                combined = (pd.concat([standalone, q4], ignore_index=True)
                              .sort_values("available_date")
                              .drop_duplicates("period_end", keep="first")
                              .reset_index(drop=True))
            else:
                combined = standalone

            if not combined.empty:
                frames[label] = combined.set_index("period_end")[["available_date", label]]

        else:
            # Balance sheet: 10-Q and 10-K, any quarterly or annual fp
            bal = raw.copy()
            if "fp" in bal.columns:
                bal = bal[bal["fp"].isin({"Q1", "Q2", "Q3", "Q4", "FY", "CY"})].copy()
            bal = (bal.sort_values("available_date")
                      .drop_duplicates("period_end", keep="first")
                      .reset_index(drop=True))
            if not bal.empty:
                frames[label] = bal.set_index("period_end")[["available_date", label]]

    if not frames:
        return None

    # Anchor on net_income; forward-fill balance sheet; outer-join other flows
    anchor = "net_income" if "net_income" in frames else next(iter(frames))
    merged = frames[anchor][["available_date", anchor]].copy()

    for label, df in frames.items():
        if label == anchor:
            continue
        if label in BALANCE_ITEMS:
            bal = df[label].sort_index()
            merged[label] = merged.index.map(
                lambda d, b=bal: b.asof(d) if len(b) > 0 and d >= b.index[0] else np.nan
            )
        else:
            merged = merged.join(df[[label]], how="outer")

    merged = merged.reset_index()
    merged.insert(0, "ticker", ticker)

    merged = merged[merged["available_date"].notna()]

    lag = (merged["available_date"] - merged["period_end"]).dt.days
    merged = merged[lag.between(0, 400)]

    if "operating_cf" in merged.columns and "capex" in merged.columns:
        merged["free_cash_flow"] = merged["operating_cf"] - merged["capex"].abs()

    merged = merged[merged["period_end"] >= "2009-01-01"]
    merged = merged.sort_values("period_end").reset_index(drop=True)

    return merged


# ── Main runner ───────────────────────────────────────────────────────────────

def run(tickers: list = None, delay: float = 0.12):
    """Pull SEC fundamentals for all S&P 500 tickers."""
    os.makedirs(OUT_DIR, exist_ok=True)

    if tickers is None:
        path = os.path.join(ROOT, "data", "sp500_constituents.csv")
        tickers = pd.read_csv(path)["Symbol"].str.replace(".", "-", regex=False).tolist()

    print("Fetching CIK map from SEC...")
    cik_map = get_cik_map()

    ok = skipped = errors = 0
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        out_path = os.path.join(OUT_DIR, f"{ticker}.csv")
        if os.path.exists(out_path):
            ok += 1
            continue

        sec_ticker = ticker.replace("-", ".")
        cik = cik_map.get(sec_ticker) or cik_map.get(ticker)

        if not cik:
            print(f"  [{i+1}/{total}] {ticker}: no CIK found")
            skipped += 1
            continue

        try:
            df = pull_ticker(ticker, cik)
            if df is not None and not df.empty:
                df.to_csv(out_path, index=False)
                ok += 1
                if (i + 1) % 25 == 0:
                    print(f"  [{i+1}/{total}] {ok} saved, {skipped} skipped, {errors} errors")
            else:
                skipped += 1
        except Exception as e:
            print(f"  [{i+1}/{total}] {ticker}: {e}")
            errors += 1

        time.sleep(delay)

    print(f"\nDone. {ok} saved | {skipped} skipped | {errors} errors")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*")
    args = parser.parse_args()
    run(tickers=args.tickers)
