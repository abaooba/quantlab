"""RSI mean-reversion — buy washed-out dips, sell into strength.

Uses Wilder's RSI (exponential smoothing with α = 1/period, his original
1978 construction). Entry: RSI crosses back **up** through the oversold
level — the knife has stopped falling, buying the *turn* rather than the
fall. Exit: RSI crosses up through the overbought level — strength has
reverted past neutral into stretched territory.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import ParamSpec, close_series, register_strategy, stateful_signal


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, Wilder-smoothed, NaN during warm-up."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    rsi = rsi.where(avg_loss != 0, 100.0)  # no losses in window → maximal strength
    rsi = rsi.where((avg_gain != 0) | (avg_loss != 0), 50.0)  # dead-flat prices → neutral
    return rsi


@register_strategy(
    "RSI Mean-Reversion",
    params=(
        ParamSpec("period", "RSI period (days)", 2, 60, 14, help="Wilder's classic is 14"),
        ParamSpec("oversold", "Oversold entry level", 5, 45, 30, help="Enter when RSI recrosses this from below"),
        ParamSpec("overbought", "Overbought exit level", 55, 95, 70, help="Exit when RSI crosses above this"),
    ),
    description=(
        "Mean reversion: buy when selling pressure exhausts (RSI turns up through the oversold "
        "line), exit once the bounce reaches overbought. Profits in range-bound markets; "
        "suffers in persistent downtrends, where 'oversold' keeps getting more oversold."
    ),
)
def signal_rsi_reversion(
    prices: pd.DataFrame | pd.Series, period: int = 14, oversold: float = 30.0, overbought: float = 70.0
) -> pd.Series:
    period = int(period)
    if not 0 < oversold < overbought < 100:
        raise ValueError("need 0 < oversold < overbought < 100")

    close = close_series(prices)
    rsi = wilder_rsi(close, period)

    prev = rsi.shift(1)
    entries = (rsi > oversold) & (prev <= oversold)
    exits = (rsi > overbought) & (prev <= overbought)

    sig = stateful_signal(close.index, entries, exits)
    sig[rsi.isna()] = 0.0  # warm-up
    return sig
