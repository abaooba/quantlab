"""In-sample / out-of-sample evaluation — the honesty core.

The split is **chronological, never shuffled**: with time series, a random
train/test split would let the model "train" on Wednesday and be "tested" on
the previous Monday — information from the future leaking backwards. Here
the first ``train_frac`` of the date range is in-sample (where you tuned
your parameters, wittingly or not) and the rest is out-of-sample (the
closest a backtest gets to the future).

Signals are computed **once over the full series** and the *results* are
split by date. That is deliberate and leak-free: every indicator here is
causal (rolling windows look backward only), so the signal at an
out-of-sample date uses only information available at that date. Computing
segments independently would instead punch a warm-up hole in the
out-of-sample segment (a 200-day MA undefined for its first 200 days).
What makes the second segment "out-of-sample" isn't how the indicator is
computed — it's that you didn't *choose your parameters* by looking at it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
import plotly.graph_objects as go

from src.engine import run_backtest
from src.metrics import drawdown_series, summarize_returns
from src.strategies.base import STRATEGY_REGISTRY
from src.style import (
    BENCHMARK_COLOR,
    STRATEGY_COLOR,
    base_layout,
    shade_out_of_sample,
)


def split_in_out_sample(
    prices: pd.DataFrame, train_frac: float = 0.7, split_date=None
) -> pd.Timestamp:
    """Return the first out-of-sample date for a chronological split.

    Either give ``train_frac`` (default: first 70% of bars are in-sample) or
    an explicit ``split_date``, which is snapped to the first bar at/after it.
    """
    n = len(prices)
    if split_date is not None:
        ts = pd.Timestamp(split_date)
        after = prices.index[prices.index >= ts]
        if len(after) < 2 or (prices.index < ts).sum() < 2:
            raise ValueError(f"split_date {ts.date()} leaves fewer than 2 bars on one side")
        return after[0]
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    cut = int(n * train_frac)
    if cut < 2 or n - cut < 2:
        raise ValueError(f"{n} bars is too few for a {train_frac:.0%} split")
    return prices.index[cut]


@dataclass
class EvaluationResult:
    strategy_name: str
    params: dict
    split_date: pd.Timestamp
    results: pd.DataFrame  # full-period strategy backtest
    benchmark: pd.DataFrame  # full-period buy-and-hold backtest
    in_sample: dict[str, float]
    out_of_sample: dict[str, float]
    benchmark_in_sample: dict[str, float] = field(default_factory=dict)
    benchmark_out_of_sample: dict[str, float] = field(default_factory=dict)
    overfit: bool = False
    verdict: str = ""


def _segment_metrics(
    results: pd.DataFrame, mask, initial_capital: float, rf: float, base_date=None
) -> dict[str, float]:
    """Metrics for one segment, with equity re-based so CAGR/DD are segment-local.

    ``base_date`` (the last bar before the segment) anchors the segment's
    starting capital so the first out-of-sample day's return isn't dropped.
    """
    seg = results.loc[mask]
    return summarize_returns(
        seg["daily_return"], positions=seg["position"],
        initial_capital=initial_capital, rf=rf, base_date=base_date,
    )


def overfitting_verdict(in_sample: dict, out_of_sample: dict) -> tuple[bool, str]:
    """Compare in/out-of-sample Sharpe and say the quiet part out loud."""
    is_s, oos_s = in_sample.get("Sharpe"), out_of_sample.get("Sharpe")
    if is_s is None or oos_s is None or pd.isna(is_s) or pd.isna(oos_s):
        return False, "➖ Not enough data in one of the segments to judge overfitting."
    if is_s > 0 and oos_s < 0:
        return True, (
            f"🚨 Overfitting red flag: profitable in-sample (Sharpe {is_s:.2f}) but "
            f"LOSES money out-of-sample (Sharpe {oos_s:.2f}). The in-sample edge was "
            "probably fit to that specific stretch of history."
        )
    if is_s > 0.5 and oos_s < is_s / 2:
        return True, (
            f"⚠️ Possible overfitting: in-sample Sharpe {is_s:.2f} vs out-of-sample "
            f"{oos_s:.2f} — more than half the apparent edge evaporates on unseen data."
        )
    if is_s <= 0 and oos_s <= 0:
        return False, (
            f"➖ No edge in either segment (Sharpe {is_s:.2f} in-sample, {oos_s:.2f} "
            "out-of-sample). Honest, at least."
        )
    if is_s <= 0 < oos_s:
        return False, (
            f"🍀 Inverse surprise: no edge in-sample (Sharpe {is_s:.2f}) yet positive "
            f"out-of-sample ({oos_s:.2f}). That's luck or a regime change, not evidence "
            "of skill — nobody would have traded this after seeing the in-sample result."
        )
    return False, (
        f"✅ In-sample (Sharpe {is_s:.2f}) and out-of-sample ({oos_s:.2f}) performance "
        "are broadly consistent — no overfitting signature on this split."
    )


def evaluate_strategy(
    strategy: str | Callable[..., pd.Series],
    prices: pd.DataFrame,
    train_frac: float = 0.7,
    cost_bps: float = 5.0,
    initial_capital: float = 100_000.0,
    rf: float = 0.0,
    split_date=None,
    **params,
) -> EvaluationResult:
    """Backtest a strategy and report in-sample vs out-of-sample metrics.

    ``strategy`` is a registered display name (see ``STRATEGY_REGISTRY``) or
    a signal function following the strategy contract. The split comes from
    ``train_frac`` unless an explicit ``split_date`` is given. A buy-and-hold
    benchmark over the same window is evaluated alongside, split identically.
    """
    if isinstance(strategy, str):
        if strategy not in STRATEGY_REGISTRY:
            raise KeyError(f"unknown strategy {strategy!r}; registered: {list(STRATEGY_REGISTRY)}")
        name, fn = strategy, STRATEGY_REGISTRY[strategy]
    else:
        name, fn = getattr(strategy, "__name__", "custom"), strategy

    split_date = split_in_out_sample(prices, train_frac, split_date=split_date)

    signals = fn(prices, **params)
    results = run_backtest(prices, signals, initial_capital=initial_capital, cost_bps=cost_bps)

    hold = pd.Series(1.0, index=prices.index)
    benchmark = run_backtest(prices, hold, initial_capital=initial_capital, cost_bps=cost_bps)

    is_mask = results.index < split_date
    oos_mask = ~is_mask
    last_is_date = results.index[is_mask][-1]  # anchors the OOS segment's base equity
    in_sample = _segment_metrics(results, is_mask, initial_capital, rf)
    out_of_sample = _segment_metrics(results, oos_mask, initial_capital, rf, base_date=last_is_date)
    bench_in = _segment_metrics(benchmark, is_mask, initial_capital, rf)
    bench_out = _segment_metrics(benchmark, oos_mask, initial_capital, rf, base_date=last_is_date)

    overfit, verdict = overfitting_verdict(in_sample, out_of_sample)

    return EvaluationResult(
        strategy_name=name,
        params=dict(params),
        split_date=split_date,
        results=results,
        benchmark=benchmark,
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        benchmark_in_sample=bench_in,
        benchmark_out_of_sample=bench_out,
        overfit=overfit,
        verdict=verdict,
    )


_PERCENT_METRICS = {"Total return", "CAGR", "Volatility (ann.)", "Max drawdown", "Win rate (daily)", "Exposure"}


def format_metric(name: str, value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    if value == float("inf"):
        return "∞"
    if name in _PERCENT_METRICS:
        return f"{value:+.1%}" if name in ("Total return", "CAGR") else f"{value:.1%}"
    return f"{value:.2f}"


def comparison_table(result: EvaluationResult) -> pd.DataFrame:
    """Metric × segment table (strategy and benchmark side by side), formatted."""
    cols = {
        "Strategy · in-sample": result.in_sample,
        "Strategy · out-of-sample": result.out_of_sample,
        "Buy & hold · in-sample": result.benchmark_in_sample,
        "Buy & hold · out-of-sample": result.benchmark_out_of_sample,
    }
    metrics = list(result.in_sample)
    data = {col: [format_metric(m, vals.get(m)) for m in metrics] for col, vals in cols.items()}
    return pd.DataFrame(data, index=pd.Index(metrics, name="Metric"))


def plot_equity_curve(result: EvaluationResult, log_scale: bool = False) -> go.Figure:
    """Strategy vs buy-and-hold equity, out-of-sample region shaded."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.benchmark.index, y=result.benchmark["equity"],
            name="Buy & hold", mode="lines",
            line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.results.index, y=result.results["equity"],
            name=result.strategy_name, mode="lines",
            line=dict(color=STRATEGY_COLOR, width=2),
        )
    )
    fig.update_layout(**base_layout(f"Equity curve — {result.strategy_name}", y_title="Equity ($)"))
    if log_scale:
        fig.update_yaxes(type="log")
    shade_out_of_sample(fig, result.split_date, result.results.index[-1])
    return fig


def plot_drawdown(result: EvaluationResult) -> go.Figure:
    """Peak-to-trough drawdown over time — the pain, not just the gain."""
    strat_dd = drawdown_series(result.results["equity"])
    bench_dd = drawdown_series(result.benchmark["equity"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bench_dd.index, y=bench_dd, name="Buy & hold", mode="lines",
            line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=strat_dd.index, y=strat_dd, name=result.strategy_name, mode="lines",
            line=dict(color=STRATEGY_COLOR, width=2),
            fill="tozeroy", fillcolor="rgba(42, 120, 214, 0.15)",
        )
    )
    fig.update_layout(**base_layout("Drawdown", y_title="Below high-water mark"))
    fig.update_yaxes(tickformat=".0%")
    shade_out_of_sample(fig, result.split_date, result.results.index[-1])
    return fig
