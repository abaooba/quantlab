"""Moving-average crossover — the canonical trend-following rule.

Long while the fast SMA sits above the slow SMA (recent prices outrunning
the longer-run average ⇒ uptrend), flat — or short, if ``allow_short`` —
otherwise. Signals are 0 during the slow window's warm-up, where the SMA is
undefined: a real trader on day 30 of a 50-day average simply doesn't have
the indicator yet.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.base import ParamSpec, close_series, register_strategy


@register_strategy(
    "MA Crossover",
    params=(
        ParamSpec("fast", "Fast SMA window (days)", 2, 150, 20, help="Short lookback — reacts quickly, whipsaws often"),
        ParamSpec("slow", "Slow SMA window (days)", 5, 300, 50, help="Long lookback — the trend benchmark"),
        ParamSpec("allow_short", "Short when fast < slow", 0, 1, 0, kind="bool",
                  help="Off = long/flat (default). On = always in the market, long or short."),
    ),
    description=(
        "Trend following: hold the asset while its short-term average trades above its "
        "long-term average. Profits from sustained trends; bleeds small losses in sideways chop."
    ),
)
def signal_ma_crossover(
    prices: pd.DataFrame | pd.Series, fast: int = 20, slow: int = 50, allow_short: bool = False
) -> pd.Series:
    fast, slow = int(fast), int(slow)
    if fast < 1 or slow < 2:
        raise ValueError("windows must be positive")
    if fast >= slow:
        raise ValueError(f"fast window ({fast}) must be shorter than slow window ({slow})")

    close = close_series(prices)
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()

    sig = pd.Series(0.0, index=close.index)
    sig[fast_ma > slow_ma] = 1.0
    if allow_short:
        sig[fast_ma < slow_ma] = -1.0
    sig[slow_ma.isna()] = 0.0  # indicator not yet defined → stay out
    return sig
