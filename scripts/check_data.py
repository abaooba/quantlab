"""Data sanity check: pull SPY and AAPL 2015–2025, flag gaps, plot closes.

Run from the repo root:  python scripts/check_data.py [--show]

Writes interactive HTML plots to assets/ (open in a browser). Gaps longer
than 4 calendar days between consecutive bars are listed; ordinary weekends
(3-day gaps around Mon/Fri) are expected and not flagged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go

from src.data import fetch_prices

TICKERS = ["SPY", "AAPL"]
START, END = "2015-01-01", "2025-01-01"
# Longest routine market closure is a 4-day holiday weekend.
MAX_ORDINARY_GAP_DAYS = 4


def check_ticker(ticker: str) -> pd.DataFrame | None:
    df = fetch_prices(ticker, START, END)
    if df is None:
        print(f"[FAIL] {ticker}: no data returned")
        return None

    print(f"[OK]   {ticker}: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")

    n_nan = int(df[["Open", "High", "Low", "Close"]].isna().sum().sum())
    print(f"       NaNs in OHLC: {n_nan}")

    gaps = df.index.to_series().diff().dt.days
    long_gaps = gaps[gaps > MAX_ORDINARY_GAP_DAYS]
    if long_gaps.empty:
        print(f"       no gaps > {MAX_ORDINARY_GAP_DAYS} calendar days")
    else:
        print(f"       {len(long_gaps)} gap(s) > {MAX_ORDINARY_GAP_DAYS} days (check vs. market holidays):")
        for date, days in long_gaps.items():
            print(f"         {date.date()}  ({int(days)} days since previous bar)")

    # ~252 trading days per year; warn if a calendar year is materially short.
    per_year = df.groupby(df.index.year).size()
    for year, count in per_year.items():
        full_year = year not in (df.index[0].year, df.index[-1].year)
        if full_year and count < 248:
            print(f"       [WARN] {year}: only {count} bars (expected ~252)")
    return df


def main() -> int:
    show = "--show" in sys.argv
    assets = Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(exist_ok=True)

    ok = True
    for ticker in TICKERS:
        df = check_ticker(ticker)
        if df is None:
            ok = False
            continue
        fig = go.Figure(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=ticker))
        fig.update_layout(
            title=f"{ticker} adjusted close, {START} → {END}",
            xaxis_title="Date",
            yaxis_title="Adjusted close ($)",
            template="plotly_white",
        )
        out = assets / f"data_check_{ticker}.html"
        fig.write_html(out, include_plotlyjs="cdn")
        print(f"       plot → {out.relative_to(assets.parent)}")
        if show:
            fig.show()

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
