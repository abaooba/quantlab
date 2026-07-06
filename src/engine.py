"""Vectorized backtest core: signals → positions → returns → equity.

The two honesty guarantees of the whole project live here:

1. **No look-ahead.** A signal computed from day *t*'s close cannot earn day
   *t*'s return — you couldn't have traded on information you didn't have
   yet. ``run_backtest`` therefore shifts signals forward one bar: the
   position held during day *t* is the signal emitted at day *t-1*'s close.

2. **No free trading.** Every unit of turnover (|Δposition|) is charged
   ``cost_bps`` basis points against that day's return. A 0→1 entry costs
   one unit; a 1→-1 reversal costs two (close the long, open the short).

Signal convention: a ``pandas.Series`` aligned to the price index with
values in {-1, 0, 1} = short / flat / long. Long-only strategies simply
never emit -1. Fractional values in [-1, 1] (partial sizing) also work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VALID_SIGNAL_RANGE = (-1.0, 1.0)


def _close_series(prices: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(prices, pd.Series):
        close = prices
    else:
        if "Close" not in prices.columns:
            raise ValueError("prices DataFrame must have a 'Close' column")
        close = prices["Close"]
    close = close.astype(float)
    if close.isna().any():
        raise ValueError("prices contain NaN closes — clean the data first")
    if (close <= 0).any():
        raise ValueError("prices must be strictly positive")
    if not close.index.is_monotonic_increasing:
        raise ValueError("price index must be sorted ascending")
    return close


def run_backtest(
    prices: pd.DataFrame | pd.Series,
    signals: pd.Series,
    initial_capital: float = 100_000.0,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Simulate trading ``signals`` over ``prices``.

    Parameters
    ----------
    prices : DataFrame with a ``Close`` column (or a close Series).
    signals : Series in {-1, 0, 1} aligned to the price index. The value at
        date *t* is the stance decided at *t*'s close, so it takes effect —
        and earns returns — from bar *t+1* onward (the look-ahead guard).
    initial_capital : starting equity in dollars.
    cost_bps : one-way transaction cost in basis points, charged per unit of
        turnover. 5 bps ≈ commission-free retail trading of liquid US ETFs
        (spread + slippage); raise it for small caps or faster strategies.

    Returns
    -------
    DataFrame indexed like ``prices`` with columns:
        ``position``     stance actually held during the bar (post-shift)
        ``asset_return`` the instrument's simple daily return
        ``daily_return`` strategy return net of costs
        ``turnover``     |Δposition| executed at that bar's open
        ``cost``         cost drag deducted from that bar's return
        ``equity``       compounded equity, starting at ``initial_capital``
    """
    close = _close_series(prices)

    if not signals.index.equals(close.index):
        raise ValueError("signals must share the price index exactly")
    sig = signals.astype(float).fillna(0.0)
    lo, hi = VALID_SIGNAL_RANGE
    if (sig < lo).any() or (sig > hi).any():
        raise ValueError(f"signals must lie in [{lo}, {hi}] (short/flat/long)")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")

    # THE look-ahead guard. Position during bar t = signal from bar t-1.
    position = sig.shift(1).fillna(0.0)

    asset_return = close.pct_change().fillna(0.0)

    # Turnover realized when the position changes going into bar t.
    turnover = position.diff().abs().fillna(position.abs().iloc[0] if len(position) else 0.0)
    cost = turnover * (cost_bps / 10_000.0)

    daily_return = position * asset_return - cost
    if (daily_return <= -1.0).any():
        raise ValueError("a daily loss of 100%+ occurred — check inputs")

    equity = initial_capital * (1.0 + daily_return).cumprod()

    return pd.DataFrame(
        {
            "position": position,
            "asset_return": asset_return,
            "daily_return": daily_return,
            "turnover": turnover,
            "cost": cost,
            "equity": equity,
        },
        index=close.index,
    )


def run_naive_backtest_do_not_use(
    prices: pd.DataFrame | pd.Series,
    signals: pd.Series,
    initial_capital: float = 100_000.0,
) -> pd.DataFrame:
    """The classic *wrong* backtest, kept for demonstration only.

    Applies day *t*'s signal to day *t*'s own return — i.e. it trades on
    information from the very close it is reacting to — and charges nothing
    to trade. ``scripts/lookahead_demo.py`` and the engine tests use it to
    show how spectacular (and fake) the resulting equity curve is.
    """
    close = _close_series(prices)
    sig = signals.astype(float).fillna(0.0)
    asset_return = close.pct_change().fillna(0.0)
    daily_return = sig * asset_return  # no shift, no costs: two lies at once
    equity = initial_capital * (1.0 + daily_return).cumprod()
    return pd.DataFrame(
        {"position": sig, "asset_return": asset_return, "daily_return": daily_return, "equity": equity},
        index=close.index,
    )


def breakeven_cost_bps(results: pd.DataFrame) -> float:
    """The per-trade cost at which this strategy's mean daily return hits zero.

    A strategy's gross edge (mean of ``position × asset_return``) is spent at
    a rate of ``cost × mean turnover`` per day, so the breakeven is their
    ratio — the strategy's entire edge expressed in basis points per unit of
    trading. Compare it to what execution actually costs: an edge worth 3 bps
    per trade cannot be harvested at 5 bps. Negative means there is no gross
    edge to spend; NaN means the strategy never traded.
    """
    turn = float(results["turnover"].mean())
    if turn <= 0:
        return float("nan")
    gross = float((results["position"] * results["asset_return"]).mean())
    return gross / turn * 10_000.0
