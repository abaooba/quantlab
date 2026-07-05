"""Volatility-targeted position sizing — risk decides the size, not conviction.

The industry-standard discipline: hold a position inversely proportional to
recent realized volatility, so the *strategy's* risk stays near a target even
as the market's risk regime swings. In calm markets you hold full size; when
volatility doubles, you halve the position. The engine already supports
fractional positions in [-1, 1], so sizing composes with any strategy signal.

Causality: the scale applied at bar *t* uses returns through bar *t* only,
and the engine's one-bar shift then delays execution — no look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import TRADING_DAYS_PER_YEAR
from src.strategies.base import close_series


def volatility_target(
    signals: pd.Series,
    prices: pd.DataFrame | pd.Series,
    target_vol: float = 0.10,
    lookback: int = 20,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Scale a {-1, 0, 1} signal by ``target_vol / realized_vol``.

    ``target_vol`` is annualized (0.10 = 10%). Scale is capped at
    ``max_leverage`` (≤ 1 for the engine's unlevered convention: sizing here
    can only de-risk, never lever up). During the vol-estimate warm-up, or
    when prices are dead flat, the position is 0 — no estimate, no trade.
    """
    if target_vol <= 0 or lookback < 2:
        raise ValueError("target_vol must be positive and lookback ≥ 2")
    if not 0 < max_leverage <= 1.0:
        raise ValueError("max_leverage must be in (0, 1] — the engine is unlevered")

    close = close_series(prices)
    if not signals.index.equals(close.index):
        raise ValueError("signals must share the price index exactly")

    realized = close.pct_change().rolling(lookback).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    with np.errstate(divide="ignore"):
        scale = (target_vol / realized).clip(upper=max_leverage)
    scale = scale.where(np.isfinite(scale) & (realized > 1e-12), 0.0)

    return (signals.astype(float) * scale).clip(-max_leverage, max_leverage)
