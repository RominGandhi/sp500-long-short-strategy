"""
Pull historical shares outstanding for S&P 500 from yfinance.
Output: data/yfinance_shares.csv
  columns: ticker, date, shares_outstanding
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_PATH = os.path.join(ROOT, "data", "yfinance_shares.csv")


def run(tickers: list = None, delay: float = 0.1):
    if tickers is None:
        path = os.path.join(ROOT, "data", "sp500_constituents.csv")
        tickers = pd.read_csv(path)["Symbol"].str.replace(".", "-", regex=False).tolist()

    frames = []
    ok = errors = 0
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        try:
            raw = yf.Ticker(ticker).get_shares_full(start="2009-01-01")
            if raw is not None and not raw.empty:
                df = raw.reset_index()
                df.columns = ["date", "shares_outstanding"]
                df["ticker"] = ticker
                frames.append(df[["ticker", "date", "shares_outstanding"]])
                ok += 1
            else:
                errors += 1
        except Exception as e:
            print(f"  [{i+1}/{total}] {ticker}: {e}")
            errors += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] {ok} ok, {errors} errors")

        time.sleep(delay)

    if not frames:
        print("No data retrieved.")
        return

    result = pd.concat(frames, ignore_index=True)
    # Strip timezone so dates are plain date strings
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\nDone. {ok} tickers | {errors} errors → {OUT_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", help="Specific tickers (default: full S&P 500)")
    args = parser.parse_args()
    run(tickers=args.tickers)
