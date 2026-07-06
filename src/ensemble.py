"""Strategy ensembles: diversification is the one free lunch — even for rules.

Two mediocre strategies with *uncorrelated* returns combine into something
better than either: the average return survives, the volatility partially
cancels. That's the same diversification argument as holding many stocks,
applied one level up — to the strategies themselves. The catch is the
correlation: trend and mean-reversion rules on the same instrument often
disagree (good), but everything correlates in a crash (bad), so the honest
analysis shows the correlation matrix next to the combined curve.

Mechanics: signals are averaged (equal weight by default) into a fractional
position in [-1, 1], which the engine backtests like any other signal —
costs charged on the *net* position's turnover, exactly as a single account
trading the blended stance would pay them.
"""

from __future__ import annotations

import pandas as pd

from src.engine import run_backtest
from src.strategies.base import STRATEGY_REGISTRY


def combine_signals(signal_map: dict[str, pd.Series], weights: dict[str, float] | None = None) -> pd.Series:
    """Weighted-average position across strategies (weights sum to 1)."""
    if not signal_map:
        raise ValueError("need at least one signal")
    names = list(signal_map)
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}
    total = sum(weights.get(n, 0.0) for n in names)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    combined = sum(signal_map[n] * (weights.get(n, 0.0) / total) for n in names)
    return combined.clip(-1.0, 1.0)


def ensemble_backtest(
    prices: pd.DataFrame,
    strategies: list[str] | None = None,
    initial_capital: float = 100_000.0,
    cost_bps: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Backtest the equal-weight blend of registered strategies (defaults: all).

    Returns ``(ensemble_results, individual_results)`` where each value is a
    standard engine result frame, so every metric in the library applies.
    """
    strategies = strategies or list(STRATEGY_REGISTRY)
    signal_map = {name: STRATEGY_REGISTRY[name](prices) for name in strategies}
    individual = {
        name: run_backtest(prices, sig, initial_capital=initial_capital, cost_bps=cost_bps)
        for name, sig in signal_map.items()
    }
    blended = combine_signals(signal_map)
    ensemble = run_backtest(prices, blended, initial_capital=initial_capital, cost_bps=cost_bps)
    return ensemble, individual


def strategy_correlations(results_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise correlation of strategies' daily returns (active days only).

    Days where *both* strategies sat flat are excluded — a correlation
    padded with shared zeros overstates independence.
    """
    names = list(results_map)
    out = pd.DataFrame(1.0, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            # Align on the shared dates FIRST, so the <3-active-days guard
            # counts exactly the sample the correlation is computed on —
            # a union-based count would let two barely-overlapping series
            # slip through and report a spurious ±1.0.
            ra, rb = results_map[a]["daily_return"].align(results_map[b]["daily_return"], join="inner")
            active = (ra != 0) | (rb != 0)
            if active.sum() < 3:
                corr = float("nan")
            else:
                corr = float(ra[active].corr(rb[active]))
            out.loc[a, b] = out.loc[b, a] = corr
    return out
