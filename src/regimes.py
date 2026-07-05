"""Volatility-regime attribution: WHERE does a strategy make and lose money?

A full-period Sharpe averages over very different markets. Conditioning
performance on the VIX level (the option-implied 30-day volatility of the
S&P 500 — the market's "fear gauge") splits the answer: trend-followers tend
to earn in calm, trending tape and get chopped up when volatility spikes;
mean-reverters often profit from the very churn that kills trend.

Classification uses **fixed absolute thresholds** (default: calm < 15,
stressed > 25 — familiar industry round numbers), not sample quantiles, so a
day's label depends only on that day's VIX close: no look-ahead. This is a
*diagnostic* lens, not a tradable signal — the engine never sees the labels.
"""

from __future__ import annotations

import pandas as pd

from src.data import fetch_prices
from src.metrics import TRADING_DAYS_PER_YEAR, sharpe_ratio

REGIME_ORDER = ["calm", "normal", "stressed"]


def classify_vix(vix_close: pd.Series, calm_below: float = 15.0, stressed_above: float = 25.0) -> pd.Series:
    """Label each day 'calm' / 'normal' / 'stressed' by its VIX close."""
    if calm_below >= stressed_above:
        raise ValueError("calm_below must be < stressed_above")
    labels = pd.Series("normal", index=vix_close.index)
    labels[vix_close < calm_below] = "calm"
    labels[vix_close > stressed_above] = "stressed"
    labels[vix_close.isna()] = pd.NA
    return labels


def vix_regimes(prices: pd.DataFrame, calm_below: float = 15.0, stressed_above: float = 25.0,
                **fetch_kwargs) -> pd.Series | None:
    """Regime label for every bar of ``prices``, from live ^VIX data.

    Returns None when VIX data can't be fetched (offline, etc.). Missing VIX
    days inside the range are forward-filled from the prior close.
    """
    start = str(prices.index[0].date())
    end = str((prices.index[-1] + pd.Timedelta(days=1)).date())
    vix = fetch_prices("^VIX", start, end, **fetch_kwargs)
    if vix is None:
        return None
    aligned = vix["Close"].reindex(prices.index).ffill()
    return classify_vix(aligned, calm_below, stressed_above)


def regime_performance(
    returns_by_name: dict[str, pd.Series], regimes: pd.Series, rf: float = 0.0
) -> pd.DataFrame:
    """Per-regime annualized return and Sharpe for each return stream.

    ``returns_by_name`` maps a display name (e.g. "MA Crossover",
    "Buy & hold") to its daily-return Series; all must share the regime
    index. Returns a frame indexed by regime with a Days column and, per
    stream, annualized mean return and Sharpe.
    """
    rows = []
    for regime in REGIME_ORDER:
        mask = (regimes == regime).fillna(False)
        row: dict[str, object] = {"Regime": regime, "Days": int(mask.sum())}
        for name, rets in returns_by_name.items():
            seg = rets[mask.reindex(rets.index, fill_value=False)]
            row[f"{name} · ann. return"] = float(seg.mean() * TRADING_DAYS_PER_YEAR) if len(seg) else float("nan")
            row[f"{name} · Sharpe"] = sharpe_ratio(seg, rf)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Regime")
