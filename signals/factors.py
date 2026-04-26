"""
Phase 2: Factor Construction
Computes value, quality, and momentum scores for each stock at each month-end.

Key methodology rules enforced here:
  - Fundamentals are lagged via available_date (no look-ahead bias)
  - TTM metrics use rolling 4-quarter sums
  - All factors are z-scored cross-sectionally, winsorized at ±3σ
  - Sector neutralization: demean within each GICS sector
  - Composite = equal-weighted average of value + quality + momentum

Output: signals/factor_scores.csv
  columns: date, ticker, sector, value, quality, momentum, composite
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from database.db import query

OUTPUT_PATH = os.path.join(ROOT, "signals", "factor_scores.csv")


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    """Returns wide DataFrame: date (index) × ticker (columns) of close prices."""
    print("Loading prices from Neon...")
    df = query("SELECT date, ticker, close FROM prices ORDER BY date")
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="close")


def load_fundamentals() -> pd.DataFrame:
    """
    Returns all fundamentals with TTM income/FCF added.
    Sorted by (ticker, period_end) — use available_date for lag enforcement.
    """
    print("Loading fundamentals from Neon...")
    df = query("""
        SELECT ticker, period_end, available_date,
               net_income, free_cash_flow, total_assets,
               equity, shares_outstanding
        FROM fundamentals
        ORDER BY ticker, period_end
    """)
    df["period_end"]    = pd.to_datetime(df["period_end"])
    df["available_date"] = pd.to_datetime(df["available_date"])

    # TTM = rolling 4-quarter sum for flow items
    df = df.sort_values(["ticker", "period_end"])
    for col in ["net_income", "free_cash_flow"]:
        df[f"{col}_ttm"] = (
            df.groupby("ticker")[col]
              .transform(lambda x: x.rolling(4, min_periods=2).sum())
        )

    # Earnings stability: std of trailing 4Q EPS proxy (net_income per share)
    df["eps_proxy"] = df["net_income"] / df["shares_outstanding"].replace(0, np.nan)
    df["earnings_stability"] = -(
        df.groupby("ticker")["eps_proxy"]
          .transform(lambda x: x.rolling(4, min_periods=3).std())
    )

    return df


def load_sector_map() -> pd.Series:
    """Returns ticker → GICS Sector mapping from the constituents CSV."""
    path = os.path.join(ROOT, "data", "sp500_constituents.csv")
    df = pd.read_csv(path)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df.set_index("Symbol")["GICS Sector"]


def get_month_ends(prices: pd.DataFrame, start: str, end: str) -> pd.DatetimeIndex:
    """Last trading day of each month within [start, end]."""
    mask = (prices.index >= start) & (prices.index <= end)
    return prices.loc[mask].resample("ME").last().index


# ── Utilities ─────────────────────────────────────────────────────────────────

def winsorize_zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score, winsorized at ±3σ."""
    s = s.dropna()
    if len(s) < 5:
        return pd.Series(np.nan, index=s.index)
    mu, sigma = s.mean(), s.std()
    if sigma == 0:
        return pd.Series(0.0, index=s.index)
    z = (s - mu) / sigma
    return z.clip(-3, 3)


def sector_neutralize(scores: pd.Series, sectors: pd.Series) -> pd.Series:
    """
    Demean scores within each GICS sector.
    Stocks not in sector map get overall mean subtracted.
    """
    aligned = sectors.reindex(scores.index)
    result = scores.copy()
    for sector, group in scores.groupby(aligned):
        result.loc[group.index] = group - group.mean()
    # Any remaining (no sector) — subtract overall mean
    no_sector = aligned.isna()
    if no_sector.any():
        result.loc[no_sector] = scores.loc[no_sector] - scores.loc[no_sector].mean()
    return result


# ── Factor functions ──────────────────────────────────────────────────────────

def value_factor(
    date: pd.Timestamp,
    universe: list,
    price_row: pd.Series,
    fund_df: pd.DataFrame,
) -> pd.Series:
    """
    Value = equal-weighted z-score of:
      - E/P  (earnings yield = net_income_ttm / market_cap)
      - FCF/P (FCF yield proxy = fcf_ttm / market_cap)
      - B/P  (book-to-price = equity / market_cap)
    """
    # Latest available fundamentals for each ticker as of date
    avail = fund_df[fund_df["available_date"] <= date]
    latest = avail.sort_values("available_date").groupby("ticker").last()
    latest = latest.reindex(universe)

    prices = price_row.reindex(universe)
    shares = latest["shares_outstanding"]
    mktcap = (prices * shares).replace(0, np.nan)

    ep  = (latest["net_income_ttm"]   / mktcap)
    fcp = (latest["free_cash_flow_ttm"] / mktcap)
    bp  = (latest["equity"]            / mktcap)

    scores = pd.DataFrame({"ep": ep, "fcp": fcp, "bp": bp}, index=universe)
    z = scores.apply(winsorize_zscore, axis=0)
    return z.mean(axis=1).rename("value")


def quality_factor(
    date: pd.Timestamp,
    universe: list,
    fund_df: pd.DataFrame,
) -> pd.Series:
    """
    Quality = equal-weighted z-score of:
      - ROIC proxy  (net_income_ttm / total_assets)
      - GP/TA       (gross profitability proxy: net_income / total_assets, Novy-Marx)
      - Earnings stability  (negative of EPS std over trailing 4Q)
    """
    avail = fund_df[fund_df["available_date"] <= date]
    latest = avail.sort_values("available_date").groupby("ticker").last()
    latest = latest.reindex(universe)

    ta = latest["total_assets"].replace(0, np.nan)

    roic   = latest["net_income_ttm"] / ta
    gp_ta  = latest["net_income"] / ta        # single-quarter as proxy for gross profit
    e_stab = latest["earnings_stability"]

    scores = pd.DataFrame({"roic": roic, "gp_ta": gp_ta, "e_stab": e_stab}, index=universe)
    z = scores.apply(winsorize_zscore, axis=0)
    return z.mean(axis=1).rename("quality")


def momentum_factor(
    date: pd.Timestamp,
    universe: list,
    prices: pd.DataFrame,
) -> pd.Series:
    """
    Momentum = 12-1 price return (from 12 months ago to 1 month ago).
    Skips the most recent month to avoid short-term reversal.
    """
    d_1m  = date - pd.DateOffset(months=1)
    d_12m = date - pd.DateOffset(months=12)

    # Find the nearest available trading day
    def nearest(target):
        idx = prices.index[prices.index <= target]
        return prices.loc[idx[-1]] if len(idx) > 0 else None

    row_1m  = nearest(d_1m)
    row_12m = nearest(d_12m)

    if row_1m is None or row_12m is None:
        return pd.Series(np.nan, index=universe, name="momentum")

    ret = (row_1m / row_12m - 1).reindex(universe)
    return winsorize_zscore(ret).rename("momentum")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(start: str = "2021-01-01", end: str = None):
    """
    Compute factor scores for all month-ends between start and end.
    Saves to signals/factor_scores.csv and prints progress.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    prices   = load_prices()
    fund_df  = load_fundamentals()
    sectors  = load_sector_map()
    universe_df = query("SELECT DISTINCT date, ticker FROM pit_universe ORDER BY date")
    universe_df["date"] = pd.to_datetime(universe_df["date"])

    month_ends = get_month_ends(prices, start, end)
    print(f"Computing factors for {len(month_ends)} month-ends ({start} → {end})\n")

    all_rows = []

    for date in month_ends:
        # Get universe on this date
        universe = (
            universe_df[universe_df["date"] == date]["ticker"].tolist()
            or
            # Fallback: use nearest available date
            universe_df[universe_df["date"] <= date]
            .sort_values("date").groupby("ticker").last().reset_index()["ticker"].tolist()
        )

        if not universe:
            continue

        price_row = prices.loc[prices.index <= date].iloc[-1]

        v  = value_factor(date, universe, price_row, fund_df)
        q  = quality_factor(date, universe, fund_df)
        m  = momentum_factor(date, universe, prices)

        df = pd.DataFrame({"value": v, "quality": q, "momentum": m})
        df.index.name = "ticker"

        # Sector neutralize each factor
        for col in ["value", "quality", "momentum"]:
            df[col] = sector_neutralize(df[col].dropna(), sectors)

        # Composite = equal-weighted average (require at least 2 of 3 factors)
        factor_cols = ["value", "quality", "momentum"]
        df["n_factors"] = df[factor_cols].notna().sum(axis=1)
        df["composite"] = df[factor_cols].mean(axis=1)
        df.loc[df["n_factors"] < 2, "composite"] = np.nan

        # Add metadata
        df["date"]   = date
        df["sector"] = sectors.reindex(df.index)
        all_rows.append(df.reset_index())

        print(f"  {date.date()}: {len(df.dropna(subset=['composite']))} stocks scored")

    result = pd.concat(all_rows, ignore_index=True)
    result = result[["date", "ticker", "sector", "value", "quality", "momentum", "composite"]]
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result):,} rows → {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end",   default=None)
    args = parser.parse_args()
    run(start=args.start, end=args.end)
