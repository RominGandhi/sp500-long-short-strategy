"""
Phase 4: Backtest Engine
Computes monthly returns, applies transaction costs, runs factor attribution,
and produces a tear sheet + charts.

Strategy: beta-hedged long book
  long_return  = Σ weight_i × raw_return_i   (top-50 composite, equal-weighted)
  short_return = −SPY_monthly_return          (single 100% NAV SPY hedge)
  total_return = long_return + short_return

Rationale for SPY hedge vs individual shorts:
  Within-S&P-500 individual shorts drag in a sustained bull market because
  even the weakest index constituents get lifted by the tide. Replacing them
  with a single SPY short isolates the long book's factor alpha without
  paying per-stock friction on 50 short positions.

Delisting assumption:
  Prices are forward-filled. Acquisition premiums are captured; true
  bankruptcies in the S&P 500 universe are extremely rare.

Transaction cost model:
  Long side: 5 / 15 bps one-way per position entering or exiting.
  SPY hedge: held constant at 100% NAV; negligible TC (~0.5 bps), omitted.
"""

import os, sys, re, io, zipfile, requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PORTFOLIOS_PATH = os.path.join(ROOT, "portfolios", "portfolios.csv")
RETURNS_PATH    = os.path.join(ROOT, "portfolios", "monthly_returns.csv")
FF_CACHE        = os.path.join(ROOT, "portfolios", "ff_factors.csv")
CHARTS_DIR      = os.path.join(ROOT, "portfolios", "charts")

BASE_BPS   = 5.0
STRESS_BPS = 15.0


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    """Wide adjusted-close DataFrame: date × ticker."""
    data_dir = os.path.join(ROOT, "data")
    frames = []
    for entry in sorted(os.listdir(data_dir)):
        if not re.match(r"\d{4}_Q\d$", entry):
            continue
        path = os.path.join(data_dir, entry, "prices.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            frames.append(df)
    combined = pd.concat(frames)
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def load_spy(start: str, end: str) -> pd.Series:
    """Monthly SPY total-return series from yfinance."""
    import yfinance as yf
    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    close = spy["Close"]["SPY"] if isinstance(spy.columns, pd.MultiIndex) else spy["Close"]
    monthly = close.resample("ME").last()
    return monthly.pct_change().dropna().rename("benchmark")


def download_ff_factors() -> pd.DataFrame:
    """
    Fama-French 5 factors + Momentum from Ken French's data library.
    Returns monthly DataFrame (decimal), columns:
      Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
    Cached locally after first download.
    """
    if os.path.exists(FF_CACHE):
        df = pd.read_csv(FF_CACHE, index_col=0, parse_dates=True)
        if df.index.max() >= pd.Timestamp.today() - pd.DateOffset(months=3):
            return df

    def _parse(url):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        fname = next(n for n in z.namelist() if n.upper().endswith(".CSV"))
        text = z.open(fname).read().decode("latin-1").strip()
        lines = text.splitlines()
        # Find first line whose first token is a 6-digit YYYYMM integer
        start = next(
            i for i, ln in enumerate(lines)
            if ln.split(",")[0].strip().isdigit()
            and len(ln.split(",")[0].strip()) == 6
        )
        # End at first blank line or non-numeric token after start
        end = len(lines)
        for i in range(start + 1, len(lines)):
            tok = lines[i].split(",")[0].strip()
            if not tok.isdigit():
                end = i
                break
        return pd.read_csv(
            io.StringIO("\n".join(lines[start:end])),
            header=None,
            names=["date"] + lines[start - 1].split(",")[1:],
        )

    ff5 = _parse("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                 "F-F_Research_Data_5_Factors_2x3_CSV.zip")
    mom = _parse("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                 "F-F_Momentum_Factor_CSV.zip")

    for df in [ff5, mom]:
        df.columns = [c.strip() for c in df.columns]
        df["date"] = (pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m")
                      + pd.offsets.MonthEnd(0))
        df.set_index("date", inplace=True)
        df[:] = df.apply(pd.to_numeric, errors="coerce")

    mom = mom.rename(columns={[c for c in mom.columns if c.strip()][0]: "Mom"})

    factors = ff5.join(mom[["Mom"]], how="left") / 100
    factors.to_csv(FF_CACHE)
    return factors


# ── Return computation ────────────────────────────────────────────────────────

def compute_returns(portfolios: pd.DataFrame,
                    prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly long/short/total gross returns.
    Uses close-to-close between consecutive formation dates.
    """
    # Pre-compute: map each date to the last available trading day
    prices_ff = prices.ffill()
    pidx = prices_ff.index

    def last_td(date):
        mask = pidx[pidx <= date]
        return mask[-1] if len(mask) > 0 else None

    formation_dates = sorted(portfolios["formation_date"].unique())
    exit_map = {t: t1 for t, t1 in zip(formation_dates[:-1], formation_dates[1:])}

    # Pre-fetch one price row per unique date (fast dict lookup later)
    unique_dates = set(formation_dates) | set(exit_map.values())
    price_rows = {d: prices_ff.loc[last_td(d)] for d in unique_dates if last_td(d) is not None}

    port = portfolios[portfolios["formation_date"].isin(exit_map)].copy()
    port["exit_date"] = port["formation_date"].map(exit_map)

    def get_px(date, ticker):
        row = price_rows.get(date)
        if row is None:
            return np.nan
        return row.get(ticker, np.nan)

    port["p_entry"] = [get_px(r.formation_date, r.ticker) for r in port.itertuples()]
    port["p_exit"]  = [get_px(r.exit_date,      r.ticker) for r in port.itertuples()]

    # Raw return; flat (0) if ticker stopped trading (forward-filled handles this)
    valid = port["p_entry"].notna() & (port["p_entry"] != 0)
    port["raw_ret"] = np.where(valid, port["p_exit"] / port["p_entry"] - 1, np.nan)

    port["contrib"] = port["weight"] * port["raw_ret"]

    # All positions are long; SPY short is added in run() after download
    def agg(grp):
        return pd.Series({
            "long_return": grp["contrib"].sum(),
            "n_long": grp["raw_ret"].notna().sum(),
        })

    monthly = (port.drop(columns=["exit_date"])
                  .groupby(port["exit_date"])
                  .apply(agg))
    monthly.index.name = "date"
    return monthly.sort_index()


def compute_tc(portfolios: pd.DataFrame, cost_bps: float) -> pd.Series:
    """
    TC per period = long-side turnover × 2% weight × cost_bps / 10000.
    SPY hedge is held constant so its TC is omitted (~0.5 bps, negligible).
    """
    dates = sorted(portfolios["formation_date"].unique())
    exit_map = {t: t1 for t, t1 in zip(dates[:-1], dates[1:])}

    costs = {}
    prev_l = set()

    for t, t1 in exit_map.items():
        curr_l = set(portfolios[portfolios["formation_date"] == t]["ticker"])
        n_trades = len(curr_l - prev_l) + len(prev_l - curr_l)
        costs[t1] = n_trades * 0.02 * cost_bps / 10_000
        prev_l = curr_l

    return pd.Series(costs, name="tc")


# ── Performance metrics ───────────────────────────────────────────────────────

def _stats(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 3:
        return {}
    total       = (1 + r).prod() - 1
    ann_ret     = (1 + total) ** (12 / n) - 1
    ann_vol     = r.std() * np.sqrt(12)
    sharpe      = ann_ret / ann_vol if ann_vol > 0 else np.nan
    downside    = r[r < 0].std() * np.sqrt(12)
    sortino     = ann_ret / downside if downside > 0 else np.nan
    nav         = (1 + r).cumprod()
    max_dd      = (nav / nav.cummax() - 1).min()
    calmar      = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    hit_rate    = (r > 0).mean()
    wins        = r[r > 0]
    losses      = r[r < 0]
    win_loss    = wins.mean() / abs(losses.mean()) if len(losses) > 0 else np.nan
    return dict(total=total, ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                sortino=sortino, max_dd=max_dd, calmar=calmar, hit_rate=hit_rate,
                best=r.max(), worst=r.min(), win_loss=win_loss, n=n)


def compute_metrics(monthly: pd.DataFrame, benchmark: pd.Series) -> dict:
    r_gross  = monthly["total_return"]
    r_net    = monthly["total_return"] - monthly["tc_base"]
    r_stress = monthly["total_return"] - monthly["tc_stress"]

    m = {
        "gross":       _stats(r_gross),
        "net_base":    _stats(r_net),
        "net_stress":  _stats(r_stress),
        "long_book":   _stats(monthly["long_return"]),
        "short_book":  _stats(monthly["short_return"]),
        "benchmark":   _stats(benchmark.reindex(monthly.index).dropna()),
        "annual":      (r_net.groupby(r_net.index.year)
                             .apply(lambda x: (1 + x).prod() - 1)
                             .to_dict()),
    }
    return m


def factor_regression(r: pd.Series, ff: pd.DataFrame) -> dict:
    """OLS on FF5 + Momentum. Returns α, t-stats, betas, R²."""
    ff_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"] if c in ff.columns]
    df = pd.concat([r.rename("r"), ff[ff_cols + (["RF"] if "RF" in ff.columns else [])]], axis=1).dropna()
    if len(df) < 24:
        return {}

    rf = df["RF"] if "RF" in df.columns else 0
    y = (df["r"] - rf).values
    X = np.column_stack([np.ones(len(y)), df[ff_cols].values])
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    yhat  = X @ b
    n, k  = X.shape
    s2    = ((y - yhat) ** 2).sum() / (n - k)
    se    = np.sqrt(s2 * np.linalg.inv(X.T @ X).diagonal())
    t     = b / se
    r2    = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    return dict(
        alpha=b[0] * 12,            # annualised
        alpha_monthly=b[0],
        alpha_tstat=t[0],
        betas=dict(zip(ff_cols, b[1:])),
        tstats=dict(zip(ff_cols, t[1:])),
        r2=r2, n=n,
    )


# ── Tearsheet ─────────────────────────────────────────────────────────────────

def print_tearsheet(m: dict, reg: dict):
    P = lambda x: f"{x*100:+7.2f}%" if not np.isnan(x) else "      —"
    F = lambda x: f"{x:7.3f}"       if not np.isnan(x) else "      —"

    print("\n" + "═" * 68)
    print("  STRATEGY TEAR SHEET")
    print("═" * 68)
    print(f"\n  {'':30} {'Gross':>8}  {'Net 5bp':>8}  {'Net 15bp':>9}  {'SPY':>6}")
    print("  " + "─" * 64)

    rows = [
        ("Total Return",    "total",    P),
        ("Ann. Return",     "ann_ret",  P),
        ("Ann. Volatility", "ann_vol",  P),
        ("Sharpe Ratio",    "sharpe",   F),
        ("Sortino Ratio",   "sortino",  F),
        ("Max Drawdown",    "max_dd",   P),
        ("Calmar Ratio",    "calmar",   F),
        ("Hit Rate",        "hit_rate", P),
        ("Best Month",      "best",     P),
        ("Worst Month",     "worst",    P),
        ("Win/Loss Ratio",  "win_loss", F),
    ]
    for name, key, fmt in rows:
        vals = [fmt(m.get(s, {}).get(key, np.nan))
                for s in ["gross", "net_base", "net_stress", "benchmark"]]
        print(f"  {name:<30} {'  '.join(vals)}")

    print("\n  LONG BOOK & SPY HEDGE")
    for label, key in [("Long book", "long_book"), ("SPY hedge (−)", "short_book")]:
        d = m.get(key, {})
        print(f"  {label:<18} Ann return: {d.get('ann_ret',0)*100:+.2f}%  "
              f"Sharpe: {d.get('sharpe',0):.2f}  "
              f"Hit rate: {d.get('hit_rate',0)*100:.0f}%")

    print("\n  ANNUAL RETURNS  (net 5 bp)")
    for yr, ret in sorted(m.get("annual", {}).items()):
        bar = ("▓" if ret >= 0 else "░") * min(int(abs(ret) * 120), 40)
        print(f"  {yr}  {ret*100:+6.1f}%  {bar}")

    if reg:
        print("\n  FACTOR ATTRIBUTION  (FF5 + Momentum)")
        sign = "✓" if abs(reg["alpha_tstat"]) > 2 else ("~" if abs(reg["alpha_tstat"]) > 1.5 else "○")
        print(f"  {sign} Alpha: {reg['alpha']*100:+.3f}% p.a.  "
              f"t-stat: {reg['alpha_tstat']:.2f}  "
              f"R²: {reg['r2']:.3f}  (n={reg['n']})")
        print(f"  {'Factor':<10} {'Beta':>7}  {'t-stat':>7}")
        print("  " + "─" * 28)
        for f, b in reg["betas"].items():
            print(f"  {f:<10} {b:>7.3f}  {reg['tstats'][f]:>7.2f}")

    print("\n" + "═" * 68)


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_charts(monthly: pd.DataFrame, benchmark: pd.Series):
    os.makedirs(CHARTS_DIR, exist_ok=True)

    r_gross  = monthly["total_return"]
    r_net    = monthly["total_return"] - monthly["tc_base"]
    r_bench  = benchmark.reindex(monthly.index).fillna(0)

    nav_gross = (1 + r_gross).cumprod()
    nav_net   = (1 + r_net).cumprod()
    nav_bench = (1 + r_bench).cumprod()

    def drawdown(nav):
        return nav / nav.cummax() - 1

    # 1. Cumulative returns
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(nav_gross, label="Strategy (gross)", color="#2196F3", lw=1.5)
    ax.plot(nav_net,   label="Strategy (net 5 bp)", color="#4CAF50", lw=1.5)
    ax.plot(nav_bench, label="SPY", color="#9E9E9E", lw=1.2, ls="--")
    ax.set_title("Cumulative Returns  (growth of $1)", fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "cumulative_returns.png"), dpi=150)
    plt.close(fig)

    # 2. Drawdown
    fig, ax = plt.subplots(figsize=(12, 4))
    dd_net   = drawdown(nav_net) * 100
    dd_bench = drawdown(nav_bench) * 100
    ax.fill_between(dd_net.index, dd_net, 0, color="#F44336", alpha=0.55, label="Strategy (net)")
    ax.fill_between(dd_bench.index, dd_bench, 0, color="#9E9E9E", alpha=0.3, label="SPY")
    ax.set_title("Drawdown (%)", fontsize=13)
    ax.set_ylabel("Drawdown %"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "drawdown.png"), dpi=150)
    plt.close(fig)

    # 3. Rolling 12-month Sharpe
    fig, ax = plt.subplots(figsize=(12, 4))
    roll_sharpe = r_net.rolling(12).apply(
        lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan,
        raw=True
    )
    ax.plot(roll_sharpe, color="#9C27B0", lw=1.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="#4CAF50", lw=0.8, ls="--", label="Sharpe = 1")
    ax.set_title("Rolling 12-Month Sharpe Ratio", fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "rolling_sharpe.png"), dpi=150)
    plt.close(fig)

    # 4. Monthly heatmap
    s = r_net.copy()
    s.index = pd.to_datetime(s.index)
    heat = (s.groupby([s.index.year, s.index.month])
             .first()
             .unstack() * 100)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    heat.columns = [month_names[c - 1] for c in heat.columns]

    fig, ax = plt.subplots(figsize=(14, max(4, len(heat) * 0.45 + 1)))
    im = ax.imshow(heat.values, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5)
    ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(len(heat.index)));   ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < 3 else "white")
    plt.colorbar(im, ax=ax, label="Return %", shrink=0.8)
    ax.set_title("Monthly Returns Heatmap  (net 5 bp)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "monthly_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"  Saved 4 charts → {CHARTS_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("Loading portfolios & prices...")
    portfolios = pd.read_csv(PORTFOLIOS_PATH, parse_dates=["formation_date"])
    prices     = load_prices()

    print("Computing position returns...")
    monthly = compute_returns(portfolios, prices)

    # SPY hedge: short_return = −SPY monthly return (100% NAV hedge)
    print("Downloading SPY benchmark...")
    try:
        benchmark = load_spy(str(monthly.index.min().date()),
                             str((monthly.index.max() + pd.DateOffset(days=5)).date()))
    except Exception as e:
        print(f"  SPY download failed ({e}) — using zeros")
        benchmark = pd.Series(0.0, index=monthly.index, name="benchmark")

    spy_monthly = benchmark.reindex(monthly.index).fillna(0)
    monthly["short_return"] = -spy_monthly
    monthly["total_return"] = monthly["long_return"] + monthly["short_return"]

    print("Computing transaction costs...")
    monthly["tc_base"]   = compute_tc(portfolios, BASE_BPS).reindex(monthly.index).fillna(0)
    monthly["tc_stress"] = compute_tc(portfolios, STRESS_BPS).reindex(monthly.index).fillna(0)

    monthly.to_csv(RETURNS_PATH)
    print(f"  Saved → {RETURNS_PATH}  ({len(monthly)} monthly periods)")

    print("Downloading Fama-French factors...")
    try:
        ff = download_ff_factors()
    except Exception as e:
        print(f"  FF download failed ({e}) — skipping regression")
        ff = pd.DataFrame()

    print("Computing metrics...")
    metrics = compute_metrics(monthly, benchmark)

    r_net = monthly["total_return"] - monthly["tc_base"]
    reg   = factor_regression(r_net, ff) if not ff.empty else {}

    print_tearsheet(metrics, reg)

    print("\nGenerating charts...")
    try:
        plot_charts(monthly, benchmark)
    except Exception as e:
        print(f"  Charts failed: {e}")


if __name__ == "__main__":
    run()
