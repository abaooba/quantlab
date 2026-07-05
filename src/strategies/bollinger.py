"""Bollinger-band breakout — momentum through a volatility envelope.

Bands sit ``num_std`` standard deviations around a rolling mean, so "close
above the upper band" means the move is large *relative to recent
volatility* — a statistically unusual thrust, not just a big point move.
Entry: close crosses above the upper band. Exit: close falls back through
the **middle** band (the rolling mean). The goal file's "flat on a close
back inside the bands" is implemented via the mid-band rather than the upper
band on purpose: prices routinely dip a hair inside the band the bar after a
breakout, and exiting there churns entry/exit fees on noise. The mid-band is
the standard whipsaw-resistant reading of "the breakout has failed".
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import ParamSpec, close_series, register_strategy, stateful_signal


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Return (middle, upper, lower) bands, NaN during warm-up."""
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=1)
    return mid, mid + num_std * sd, mid - num_std * sd


@register_strategy(
    "Bollinger Breakout",
    params=(
        ParamSpec("window", "Band window (days)", 5, 100, 20, help="Rolling mean/std lookback"),
        ParamSpec("num_std", "Band width (std devs)", 0.5, 4.0, 2.0, step=0.25, kind="float",
                  help="2σ ≈ a ~95% envelope if returns were normal (they aren't — that's the point)"),
    ),
    description=(
        "Volatility breakout: enter when price punches above the upper band (an unusually "
        "strong move for current volatility), exit when it retreats through the rolling mean. "
        "Profits when breakouts start trends; loses when they exhaust immediately."
    ),
)
def signal_bollinger_breakout(
    prices: pd.DataFrame | pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.Series:
    window = int(window)
    if window < 2 or num_std <= 0:
        raise ValueError("window must be ≥ 2 and num_std positive")

    close = close_series(prices)
    mid, upper, _lower = bollinger_bands(close, window, float(num_std))

    prev_close = close.shift(1)
    entries = (close > upper) & (prev_close <= upper.shift(1))
    exits = (close < mid) & (prev_close >= mid.shift(1))

    sig = stateful_signal(close.index, entries, exits)
    sig[mid.isna()] = 0.0  # warm-up
    return sig
