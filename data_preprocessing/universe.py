"""
Step 1.1 — Point-in-time S&P 500 universe

Reconstructs which tickers were in the S&P 500 on any given date using:
  - Wikipedia's current constituents table
  - Wikipedia's full changes history (additions/removals with dates)

Output: data/universe/pit_membership.csv
  columns: date, ticker, in_index (1/0)

This eliminates survivorship bias: your backtest uses only tickers
that were actually in the index on each date, including since-delisted ones.
"""

import os
import requests
import pandas as pd
from io import StringIO
from datetime import date

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data", "universe")
WIKI_URL  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS   = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}


def fetch_wikipedia_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))

    current = tables[0].copy()
    current["Symbol"] = current["Symbol"].str.replace(".", "-", regex=False)

    changes = tables[1].copy()
    changes.columns = ["date", "added_ticker", "added_name", "removed_ticker", "removed_name", "reason"]
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    changes = changes.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    changes["added_ticker"]   = changes["added_ticker"].str.replace(".", "-", regex=False)
    changes["removed_ticker"] = changes["removed_ticker"].str.replace(".", "-", regex=False)

    return current, changes


def build_pit_membership(start: str = "2015-01-01") -> pd.DataFrame:
    """
    Returns a DataFrame with every (trading date, ticker) pair and
    whether that ticker was in the S&P 500 on that date.
    """
    print("Fetching Wikipedia constituent and changes data...")
    current_df, changes_df = fetch_wikipedia_tables()

    # Current members as of today
    current_tickers = set(current_df["Symbol"].tolist())

    # Collect every ticker that has ever appeared (current + historical)
    added_tickers   = set(changes_df["added_ticker"].dropna())
    removed_tickers = set(changes_df["removed_ticker"].dropna())
    all_tickers     = current_tickers | added_tickers | removed_tickers
    all_tickers.discard(float("nan"))
    all_tickers = {t for t in all_tickers if isinstance(t, str)}

    print(f"  Current members:    {len(current_tickers)}")
    print(f"  Ever-in-index:      {len(all_tickers)}")

    # Build a daily membership set by replaying changes backwards from today
    # Start with today's members, then walk backwards through changes
    # to reconstruct membership on any prior date.

    start_date = pd.Timestamp(start)
    end_date   = pd.Timestamp(date.today())

    # Filter changes within our window (and slightly before to anchor state)
    relevant = changes_df[changes_df["date"] <= end_date].copy()

    # Step 1: start from current members and work backwards
    members_at_end = set(current_tickers)

    # Replay changes in reverse to get members at start_date
    future_changes = relevant[relevant["date"] > start_date].sort_values("date", ascending=False)
    members_at_start = set(members_at_end)
    for _, row in future_changes.iterrows():
        # Undo an addition = remove that ticker
        if isinstance(row["added_ticker"], str):
            members_at_start.discard(row["added_ticker"])
        # Undo a removal = add that ticker back
        if isinstance(row["removed_ticker"], str):
            members_at_start.add(row["removed_ticker"])

    # Step 2: walk forward day-by-day applying changes
    # Use monthly snapshots for efficiency (interpolate daily from monthly)
    change_dates = sorted(relevant[relevant["date"] >= start_date]["date"].unique())

    snapshots = []  # list of (date, frozenset of tickers)
    current_members = set(members_at_start)
    snapshots.append((start_date, frozenset(current_members)))

    for cdate in change_dates:
        day_changes = relevant[relevant["date"] == cdate]
        for _, row in day_changes.iterrows():
            if isinstance(row["removed_ticker"], str):
                current_members.discard(row["removed_ticker"])
            if isinstance(row["added_ticker"], str):
                current_members.add(row["added_ticker"])
        snapshots.append((cdate, frozenset(current_members)))

    snapshots.append((end_date, frozenset(current_members)))

    # Step 3: build a trading-day calendar and fill forward
    print("Building daily membership table...")
    bdays = pd.bdate_range(start=start_date, end=end_date)

    snap_dates  = [s[0] for s in snapshots]
    snap_sets   = [s[1] for s in snapshots]

    rows = []
    snap_idx = 0
    for bday in bdays:
        # Advance snapshot pointer
        while snap_idx + 1 < len(snap_dates) and snap_dates[snap_idx + 1] <= bday:
            snap_idx += 1
        members = snap_sets[snap_idx]
        for ticker in members:
            rows.append((bday.date(), ticker))

    pit = pd.DataFrame(rows, columns=["date", "ticker"])
    pit["date"] = pd.to_datetime(pit["date"])

    os.makedirs(DATA_DIR, exist_ok=True)
    pit.to_csv(os.path.join(DATA_DIR, "pit_membership.csv"), index=False)

    # Also save the full ever-in-index ticker list
    all_df = pd.DataFrame(sorted(all_tickers), columns=["ticker"])
    all_df.to_csv(os.path.join(DATA_DIR, "all_tickers_ever.csv"), index=False)

    print(f"  Trading days:  {pit['date'].nunique()}")
    print(f"  Unique tickers in PIT table: {pit['ticker'].nunique()}")
    print(f"Saved to {DATA_DIR}/")
    return pit


def load_pit() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "pit_membership.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def members_on(dt) -> list[str]:
    """Return list of tickers in the S&P 500 on a given date."""
    pit = load_pit()
    dt  = pd.Timestamp(dt)
    return pit[pit["date"] == dt]["ticker"].tolist()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    args = parser.parse_args()
    build_pit_membership(start=args.start)
