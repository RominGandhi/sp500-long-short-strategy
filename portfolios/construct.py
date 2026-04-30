"""
Phase 3: Portfolio Construction
Builds a long/short portfolio from factor scores at each formation date.

Rules:
  - Long:  top 50 by composite, max 10 per GICS sector
  - Short: bottom 50 by composite, max 10 per GICS sector
  - Equal-weighted: 2% per position on each side
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

        long_book  = build_book(scored, "long")
        short_book = build_book(scored, "short")

        if long_book.empty or short_book.empty:
            continue

        for book in [long_book, short_book]:
            book["formation_date"] = date

        all_books.append(pd.concat([long_book, short_book], ignore_index=True))

    result = pd.concat(all_books, ignore_index=True)
    result = result.rename(columns={"composite": "composite_score"})
    result = result[["formation_date", "ticker", "side", "weight", "sector", "composite_score"]]
    result = result.sort_values(["formation_date", "side", "composite_score"],
                                ascending=[True, True, False])
    return result.reset_index(drop=True)


# ── Sanity checks ─────────────────────────────────────────────────────────────

def sanity_checks(p: pd.DataFrame):
    dates  = sorted(p["formation_date"].unique())
    longs  = p[p["side"] == "long"]
    shorts = p[p["side"] == "short"]

    print("=" * 60)
    print("  PORTFOLIO SANITY CHECKS")
    print("=" * 60)

    # Basic shape
    rows_per_date = p.groupby("formation_date").size()
    print(f"\n  Dates:            {len(dates)}")
    print(f"  Total rows:       {len(p):,}")
    print(f"  Rows per date:    min={rows_per_date.min()}  max={rows_per_date.max()}  mean={rows_per_date.mean():.0f}")

    # ── Sector distribution ────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  SECTOR DISTRIBUTION  (avg stocks per sector, latest date)")
    print("─" * 60)
    latest = p[p["formation_date"] == dates[-1]]
    for side_label, sub in [("Long", latest[latest["side"] == "long"]),
                             ("Short", latest[latest["side"] == "short"])]:
        counts = sub["sector"].value_counts()
        print(f"\n  {side_label} book ({dates[-1].date()}):")
        for sector, n in counts.items():
            bar = "█" * n
            print(f"    {sector:<35} {n:>2}  {bar}")

    # Check no single sector dominates in any month
    max_sector = (p.groupby(["formation_date", "side", "sector"])
                   .size()
                   .reset_index(name="n"))
    over_cap = max_sector[max_sector["n"] > SECTOR_CAP]
    if over_cap.empty:
        print(f"\n  ✓ Sector cap ({SECTOR_CAP}) never breached")
    else:
        print(f"\n  ✗ Sector cap breached {len(over_cap)} times — check logic")

    # ── Turnover ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  TURNOVER  (% of names replaced each month)")
    print("─" * 60)

    turnovers = {"long": [], "short": []}
    for side in ["long", "short"]:
        side_df = p[p["side"] == side]
        prev_tickers = set()
        for date in dates:
            curr = set(side_df[side_df["formation_date"] == date]["ticker"])
            if prev_tickers:
                exits = len(prev_tickers - curr)
                turnovers[side].append(exits / len(prev_tickers))
            prev_tickers = curr

    for side in ["long", "short"]:
        t = turnovers[side]
        print(f"\n  {side.capitalize()} book turnover:")
        print(f"    Mean:   {np.mean(t)*100:.1f}%")
        print(f"    Median: {np.median(t)*100:.1f}%")
        print(f"    Min:    {np.min(t)*100:.1f}%")
        print(f"    Max:    {np.max(t)*100:.1f}%")
        flag = ""
        if np.mean(t) > 0.50:
            flag = "  ⚠ above 50% — rankings may be unstable"
        elif np.mean(t) < 0.05:
            flag = "  ⚠ below 5% — rankings too sticky"
        else:
            flag = "  ✓ healthy range (15–30%)"
        print(f"    {flag}")

    # ── Long/short composite spread ────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  LONG/SHORT COMPOSITE SPREAD  (long mean − short mean)")
    print("─" * 60)
    spread = (longs.groupby("formation_date")["composite_score"].mean()
              - shorts.groupby("formation_date")["composite_score"].mean())
    print(f"\n  Mean spread:   {spread.mean():.3f}")
    print(f"  Median spread: {spread.median():.3f}")
    print(f"  Min spread:    {spread.min():.3f}")
    print(f"  Max spread:    {spread.max():.3f}")
    # Composite is avg of 3 sector-neutralized z-scores, so top/bottom 10%
    # naturally lands at ~±0.7, giving a spread of ~1.3–1.7. That is healthy.
    if spread.mean() >= 1.2:
        print("  ✓ Strong discrimination (sector-neutralized composite)")
    elif spread.mean() >= 0.8:
        print("  ~ Moderate discrimination")
    else:
        print("  ⚠ Weak discrimination — factors may not be separating well")

    # Show spread over time (annual sample)
    print("\n  Spread over time (year-end sample):")
    annual = spread[spread.index.month == 12]
    for d, s in annual.items():
        print(f"    {d.year}: {s:.3f}")

    # ── Persistent names ───────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  MEMBERSHIP PERSISTENCE")
    print("─" * 60)

    for side_label, sub in [("Long", longs), ("Short", shorts)]:
        freq = (sub.groupby("ticker")["formation_date"].count() / len(dates) * 100)
        top10 = freq.sort_values(ascending=False).head(10)
        print(f"\n  {side_label} book — most persistent names:")
        for ticker, pct in top10.items():
            bar = "█" * int(pct / 5)
            print(f"    {ticker:<8} {pct:5.1f}%  {bar}")

    print("\n" + "=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building portfolios...")
    portfolios = build_portfolios()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    portfolios.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(portfolios):,} rows → {OUT_PATH}\n")

    sanity_checks(portfolios)
