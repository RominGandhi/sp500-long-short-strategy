"""
Phase 3: Portfolio Construction
Builds a beta-hedged long portfolio from factor scores at each formation date.

Rules:
  - Long:  top 50 by composite, max 10 per GICS sector
  - Equal-weighted: 2% per position (= 100% gross long)
  - Market hedge: single SPY short at 100% NAV handled in backtest.py
  - Greedy sector cap: walk sorted list, skip if sector is full

Output: portfolios/portfolios.csv
  columns: formation_date, ticker, side, weight, sector, composite_score
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SCORES_PATH = os.path.join(ROOT, "signals", "factor_scores.csv")
OUT_PATH    = os.path.join(ROOT, "portfolios", "portfolios.csv")

N_SIDE     = 50
SECTOR_CAP = 10
WEIGHT     = 0.02


# ── Portfolio builder ─────────────────────────────────────────────────────────

def build_book(scored: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Greedy sector-capped selection.
    Walks the sorted list and skips any ticker whose sector already has 10 reps.
    """
    candidates = scored.sort_values("composite", ascending=(side == "short"))

    selected = []
    sector_counts: dict = {}

    for _, row in candidates.iterrows():
        if len(selected) >= N_SIDE:
            break
        sector = row["sector"] if pd.notna(row["sector"]) else "__unknown__"
        if sector_counts.get(sector, 0) < SECTOR_CAP:
            selected.append(row)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if not selected:
        return pd.DataFrame()

    out = pd.DataFrame(selected)[["ticker", "sector", "composite"]].copy()
    out["side"]   = side
    out["weight"] = WEIGHT
    return out


def build_portfolios() -> pd.DataFrame:
    scores = pd.read_csv(SCORES_PATH, parse_dates=["date"])
    scores = scores.dropna(subset=["composite"])

    all_books = []

    for date, group in scores.groupby("date"):
        scored = group[["ticker", "sector", "composite"]].copy()
        long_book = build_book(scored, "long")
        if long_book.empty:
            continue
        long_book["formation_date"] = date
        all_books.append(long_book)

    result = pd.concat(all_books, ignore_index=True)
    result = result.rename(columns={"composite": "composite_score"})
    result = result[["formation_date", "ticker", "side", "weight", "sector", "composite_score"]]
    result = result.sort_values(["formation_date", "composite_score"],
                                ascending=[True, False])
    return result.reset_index(drop=True)


# ── Sanity checks ─────────────────────────────────────────────────────────────

def sanity_checks(p: pd.DataFrame):
    dates = sorted(p["formation_date"].unique())

    print("=" * 60)
    print("  PORTFOLIO SANITY CHECKS  (long book + SPY hedge)")
    print("=" * 60)

    rows_per_date = p.groupby("formation_date").size()
    print(f"\n  Dates:            {len(dates)}")
    print(f"  Total rows:       {len(p):,}")
    print(f"  Rows per date:    min={rows_per_date.min()}  max={rows_per_date.max()}  mean={rows_per_date.mean():.0f}")

    # ── Sector distribution ────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  SECTOR DISTRIBUTION  (long book, latest date)")
    print("─" * 60)
    latest = p[p["formation_date"] == dates[-1]]
    counts = latest["sector"].value_counts()
    for sector, n in counts.items():
        bar = "█" * n
        print(f"  {sector:<35} {n:>2}  {bar}")

    max_sector = (p.groupby(["formation_date", "sector"])
                   .size()
                   .reset_index(name="n"))
    over_cap = max_sector[max_sector["n"] > SECTOR_CAP]
    if over_cap.empty:
        print(f"\n  ✓ Sector cap ({SECTOR_CAP}) never breached")
    else:
        print(f"\n  ✗ Sector cap breached {len(over_cap)} times — check logic")

    # ── Turnover ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  TURNOVER  (% of long names replaced each month)")
    print("─" * 60)
    prev_tickers = set()
    turnovers = []
    for date in dates:
        curr = set(p[p["formation_date"] == date]["ticker"])
        if prev_tickers:
            turnovers.append(len(prev_tickers - curr) / len(prev_tickers))
        prev_tickers = curr

    t = turnovers
    print(f"\n  Mean:   {np.mean(t)*100:.1f}%")
    print(f"  Median: {np.median(t)*100:.1f}%")
    print(f"  Min:    {np.min(t)*100:.1f}%")
    print(f"  Max:    {np.max(t)*100:.1f}%")
    if np.mean(t) > 0.50:
        print("  ⚠ above 50% — rankings may be unstable")
    elif np.mean(t) < 0.05:
        print("  ⚠ below 5% — rankings too sticky")
    else:
        print("  ✓ healthy range (15–30%)")

    # ── Persistent names ───────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  MOST PERSISTENT LONG NAMES")
    print("─" * 60)
    freq = (p.groupby("ticker")["formation_date"].count() / len(dates) * 100)
    for ticker, pct in freq.sort_values(ascending=False).head(10).items():
        bar = "█" * int(pct / 5)
        print(f"  {ticker:<8} {pct:5.1f}%  {bar}")

    print("\n" + "=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building portfolios...")
    portfolios = build_portfolios()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    portfolios.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(portfolios):,} rows → {OUT_PATH}\n")

    sanity_checks(portfolios)
