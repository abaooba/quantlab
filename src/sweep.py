"""Parameter sweep — making overfitting visible as a picture.

Sweep a strategy's parameter grid and record in-sample vs out-of-sample
Sharpe for every combination. Two heatmaps side by side tell the story no
single backtest can: the in-sample surface has a seductive bright peak (the
combo an optimizer would pick), and the out-of-sample surface shows how
little of that peak survives contact with unseen data. The "best" cell
moving — or turning cold — *is* overfitting, rendered directly.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.engine import run_backtest
from src.evaluate import split_in_out_sample
from src.metrics import build_equity, cagr, sharpe_ratio
from src.strategies.base import expand_grid, resolve_strategy
from src.style import DIVERGING_SCALE, base_layout

MAX_COMBOS = 900


def parameter_sweep(
    strategy: str | Callable[..., pd.Series],
    prices: pd.DataFrame,
    param_grid: dict[str, list],
    train_frac: float = 0.7,
    cost_bps: float = 5.0,
    rf: float = 0.0,
) -> pd.DataFrame:
    """One row per parameter combo: in/out-of-sample Sharpe and CAGR.

    Combos that are invalid for the strategy (e.g. fast ≥ slow for an MA
    crossover) are skipped rather than failing the sweep.
    """
    _, fn = resolve_strategy(strategy)
    combos = expand_grid(param_grid, MAX_COMBOS)

    split_date = split_in_out_sample(prices, train_frac)
    rows = []
    for combo in combos:
        try:
            signals = fn(prices, **combo)
        except ValueError:
            continue  # invalid combo for this strategy
        res = run_backtest(prices, signals, cost_bps=cost_bps)
        rets = res["daily_return"]
        is_rets, oos_rets = rets[rets.index < split_date], rets[rets.index >= split_date]
        last_is_date = is_rets.index[-1]
        rows.append(
            {
                **combo,
                "is_sharpe": sharpe_ratio(is_rets, rf),
                "oos_sharpe": sharpe_ratio(oos_rets, rf),
                "is_cagr": cagr(build_equity(is_rets)),
                "oos_cagr": cagr(build_equity(oos_rets, base_date=last_is_date)),
            }
        )
    if not rows:
        raise ValueError("no valid parameter combination in the grid")
    return pd.DataFrame(rows)


def best_in_sample(sweep_df: pd.DataFrame) -> pd.Series:
    """The combo a naive optimizer would pick (highest in-sample Sharpe)."""
    valid = sweep_df["is_sharpe"].dropna()
    if valid.empty:
        raise ValueError("every combo produced an undefined in-sample Sharpe (all flat?)")
    return sweep_df.loc[valid.idxmax()]


def oos_rank_of_is_best(sweep_df: pd.DataFrame) -> tuple[int, float]:
    """Where the in-sample winner lands out-of-sample: (rank, percentile).

    Rank 1 = best. Percentile 0.95 means it beat 95% of the *other* combos —
    a champion that ranks middling out-of-sample was curve-fit. Combos with
    an undefined out-of-sample Sharpe are excluded; a champion that is itself
    undefined out-of-sample counts as worse than every measurable combo.
    """
    best = best_in_sample(sweep_df)
    peers = sweep_df["oos_sharpe"].dropna()
    champ = best["oos_sharpe"]
    if pd.isna(champ):
        return len(peers) + 1, 0.0
    others = peers.drop(index=best.name, errors="ignore")
    rank = int((peers > champ).sum()) + 1
    pct = float((others < champ).mean()) if len(others) else 0.0
    return rank, pct


def sweep_heatmap_pair(
    sweep_df: pd.DataFrame, x: str, y: str, metric: str = "sharpe"
) -> go.Figure:
    """Side-by-side in-sample / out-of-sample heatmaps over a 2-D grid.

    Shared diverging color scale centered at 0 (red = losing, blue =
    making money), so the two panels are directly comparable. The
    in-sample champion cell is marked ★ on both panels — watch it dim.
    """
    is_col, oos_col = f"is_{metric}", f"oos_{metric}"
    is_grid = sweep_df.pivot_table(index=y, columns=x, values=is_col)
    oos_grid = sweep_df.pivot_table(index=y, columns=x, values=oos_col)

    zmax = float(np.nanmax(np.abs(np.concatenate([is_grid.to_numpy().ravel(), oos_grid.to_numpy().ravel()]))))
    zmax = zmax if np.isfinite(zmax) and zmax > 0 else 1.0

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
        subplot_titles=("In-sample (where you tuned)", "Out-of-sample (what you'd get)"),
    )
    common = dict(colorscale=DIVERGING_SCALE, zmin=-zmax, zmax=zmax, zmid=0.0)
    fig.add_trace(go.Heatmap(z=is_grid.values, x=is_grid.columns, y=is_grid.index,
                             colorbar=dict(title=metric.capitalize(), x=1.02), **common), row=1, col=1)
    fig.add_trace(go.Heatmap(z=oos_grid.values, x=oos_grid.columns, y=oos_grid.index,
                             showscale=False, **common), row=1, col=2)

    best = best_in_sample(sweep_df)
    for col in (1, 2):
        fig.add_annotation(x=best[x], y=best[y], text="★", showarrow=False,
                           font=dict(size=16, color="#0b0b0b"), row=1, col=col)

    layout = base_layout(f"Parameter sweep — {metric} by ({x}, {y})", height=440)
    layout.pop("hovermode")
    fig.update_layout(**layout)
    fig.update_xaxes(title_text=x, row=1, col=1)
    fig.update_xaxes(title_text=x, row=1, col=2)
    fig.update_yaxes(title_text=y, row=1, col=1)
    return fig
