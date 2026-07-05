"""Anchored walk-forward optimization — out-of-sample testing, industrialized.

A single 70/30 split still lets you overfit *to the split*: tweak parameters
until the one out-of-sample stretch looks good and it has quietly become
in-sample. Walk-forward closes that loophole by re-optimizing repeatedly and
only ever counting performance on data the optimizer hadn't seen:

    train on [start … t₁) → pick best params → trade them on [t₁ … t₂)
    train on [start … t₂) → re-pick        → trade on [t₂ … t₃)   …

The chained test segments form one continuous out-of-sample equity curve —
the closest a backtest gets to "how would this have felt live, honestly?"
The contrast number reported alongside is the *hindsight* Sharpe: the best
single parameter set chosen on the full sample, i.e. what an overfitter
would put on their slide deck.

Causality note: signals for each candidate are computed once over the full
series. Rolling indicators are backward-looking, so a signal value at date
*t* is identical whether computed on data through *t* or through 2035 —
what varies per window is only which *parameters* the train segment picks.
At each window seam the chained curve charges the *actual* transition cost
(from the previous window's ending position to the new run's position),
replacing whatever historical transition the new run happened to embed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.engine import run_backtest
from src.metrics import build_equity, sharpe_ratio, summarize_returns
from src.strategies.base import STRATEGY_REGISTRY
from src.style import ACCENT_COLOR, BENCHMARK_COLOR, MUTED_INK, base_layout, plot_date

MAX_COMBOS = 400


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # last train bar (window trains on [start … here])
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_sharpe: float
    test_sharpe: float


@dataclass
class WalkForwardResult:
    strategy_name: str
    param_grid: dict
    windows: list[WalkForwardWindow]
    oos_returns: pd.Series  # chained test-segment daily returns
    oos_equity: pd.Series
    oos_metrics: dict[str, float]
    hindsight_params: dict  # best single combo on the full sample…
    hindsight_sharpe: float  # …and its (in-sample-fit) Sharpe
    benchmark_metrics: dict[str, float] = field(default_factory=dict)


def _combos(param_grid: dict[str, list]) -> list[dict]:
    keys = list(param_grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]
    if len(combos) > MAX_COMBOS:
        raise ValueError(f"{len(combos)} combos > {MAX_COMBOS}; thin the grid")
    return combos


def walk_forward(
    strategy: str | Callable[..., pd.Series],
    prices: pd.DataFrame,
    param_grid: dict[str, list],
    n_windows: int = 5,
    initial_train_frac: float = 0.4,
    cost_bps: float = 5.0,
    initial_capital: float = 100_000.0,
    rf: float = 0.0,
) -> WalkForwardResult:
    """Run anchored walk-forward optimization over a parameter grid.

    The first ``initial_train_frac`` of bars seeds the first train window;
    the remainder is cut into ``n_windows`` equal test segments. Each window
    trains on *all* data before its test segment (anchored/expanding).
    """
    if isinstance(strategy, str):
        name, fn = strategy, STRATEGY_REGISTRY[strategy]
    else:
        name, fn = getattr(strategy, "__name__", "custom"), strategy
    if not 0.0 < initial_train_frac < 1.0:
        raise ValueError("initial_train_frac must be in (0, 1)")

    n = len(prices)
    first_test = int(n * initial_train_frac)
    if first_test < 30 or n - first_test < n_windows * 2:
        raise ValueError("series too short for this walk-forward configuration")
    test_chunks = np.array_split(np.arange(first_test, n), n_windows)

    # One full-series backtest per combo; every window then just slices it.
    # Combos invalid for the strategy (e.g. fast ≥ slow) are skipped.
    combos: list[dict] = []
    runs: list[pd.DataFrame] = []
    for combo in _combos(param_grid):
        try:
            signals = fn(prices, **combo)
        except ValueError:
            continue
        res = run_backtest(prices, signals, initial_capital=initial_capital, cost_bps=cost_bps)
        combos.append(combo)
        runs.append(res)
    if not runs:
        raise ValueError("no valid parameter combination in the grid")

    windows: list[WalkForwardWindow] = []
    oos_parts: list[pd.Series] = []
    best_i = 0  # fallback stance if the first window has nothing measurable
    prev_pos = 0.0  # the walk-forward trader starts flat
    for chunk in test_chunks:
        t0, t1 = int(chunk[0]), int(chunk[-1]) + 1
        train_sharpes = [sharpe_ratio(r["daily_return"].iloc[:t0], rf) for r in runs]
        if np.all(np.isnan(train_sharpes)):
            # Every combo was flat/undefined on this train window (e.g. still
            # inside indicator warm-up): keep the previous window's parameters
            # rather than crash — a live trader with no signal changes nothing.
            pass
        else:
            best_i = int(np.nanargmax(train_sharpes))
        run = runs[best_i]
        test_rets = run["daily_return"].iloc[t0:t1].copy()
        # Seam correction: the chained trader crosses the boundary holding the
        # PREVIOUS window's ending position, not this run's historical one —
        # swap the run's embedded transition cost for the actual one.
        embedded = float(run["turnover"].iloc[t0])
        actual = abs(float(run["position"].iloc[t0]) - prev_pos)
        test_rets.iloc[0] += (embedded - actual) * cost_bps / 10_000.0
        prev_pos = float(run["position"].iloc[t1 - 1])
        windows.append(
            WalkForwardWindow(
                train_start=prices.index[0],
                train_end=prices.index[t0 - 1],
                test_start=prices.index[t0],
                test_end=prices.index[t1 - 1],
                best_params=combos[best_i],
                train_sharpe=float(train_sharpes[best_i]),
                test_sharpe=sharpe_ratio(test_rets, rf),
            )
        )
        oos_parts.append(test_rets)

    oos_returns = pd.concat(oos_parts)
    base_date = prices.index[first_test - 1]  # last train-only bar anchors the curve
    oos_equity = build_equity(oos_returns, initial_capital, base_date=base_date)
    oos_metrics = summarize_returns(
        oos_returns, initial_capital=initial_capital, rf=rf, base_date=base_date
    )

    full_sharpes = [sharpe_ratio(r["daily_return"], rf) for r in runs]
    hind_i = 0 if np.all(np.isnan(full_sharpes)) else int(np.nanargmax(full_sharpes))

    bench = run_backtest(prices, pd.Series(1.0, index=prices.index),
                         initial_capital=initial_capital, cost_bps=cost_bps)
    bench_metrics = summarize_returns(
        bench.loc[oos_returns.index, "daily_return"],
        initial_capital=initial_capital, rf=rf, base_date=base_date,
    )

    return WalkForwardResult(
        strategy_name=name,
        param_grid=param_grid,
        windows=windows,
        oos_returns=oos_returns,
        oos_equity=oos_equity,
        oos_metrics=oos_metrics,
        hindsight_params=combos[hind_i],
        hindsight_sharpe=float(full_sharpes[hind_i]),
        benchmark_metrics=bench_metrics,
    )


def plot_walk_forward(result: WalkForwardResult, benchmark_equity: pd.Series | None = None) -> go.Figure:
    """Chained out-of-sample equity with window boundaries and chosen params."""
    fig = go.Figure()
    if benchmark_equity is not None:
        fig.add_trace(
            go.Scatter(x=benchmark_equity.index, y=benchmark_equity, name="Buy & hold (same window)",
                       mode="lines", line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dash"))
        )
    fig.add_trace(
        go.Scatter(x=result.oos_equity.index, y=result.oos_equity,
                   name="Walk-forward (all out-of-sample)", mode="lines",
                   line=dict(color=ACCENT_COLOR, width=2))
    )
    for w in result.windows:
        fig.add_vline(x=plot_date(w.test_start), line_dash="dot", line_color=MUTED_INK, line_width=1)
        params = ", ".join(f"{k}={v}" for k, v in w.best_params.items())
        fig.add_annotation(x=plot_date(w.test_start), y=1.04, yref="paper", xanchor="left",
                           showarrow=False, text=params, font=dict(size=10, color=MUTED_INK))
    fig.update_layout(**base_layout(
        f"Walk-forward equity — {result.strategy_name} (re-optimized each window)",
        y_title="Equity ($)",
    ))
    return fig
