"""Shared chart styling: one place for colors and plotly layout defaults.

Colors come from a CVD-validated palette. Roles, not decoration:
- the strategy is always blue; the buy-and-hold benchmark is always a muted
  gray reference line (color follows the entity across every chart);
- the out-of-sample region is a light neutral wash;
- diverging scales (Sharpe heatmaps) run red ↔ gray ↔ blue so "negative"
  reads warm, "zero" reads like nothing, "positive" reads cool.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

STRATEGY_COLOR = "#2a78d6"  # blue — the strategy, on every chart
BENCHMARK_COLOR = "#898781"  # muted gray — buy-and-hold reference
ACCENT_COLOR = "#1baf7a"  # aqua — secondary series (e.g. walk-forward curve)
NEGATIVE_COLOR = "#e34948"  # red — losses/negative pole
OOS_WASH = "rgba(137, 135, 129, 0.12)"  # out-of-sample shading
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"

# red ← gray → blue, anchored so 0 lands on the neutral midpoint
DIVERGING_SCALE = [
    (0.0, "#d03b3b"),
    (0.25, "#ec9a9a"),
    (0.5, "#f0efec"),
    (0.75, "#86b6ef"),
    (1.0, "#104281"),
]


def base_layout(title: str, *, x_title: str = "", y_title: str = "", height: int = 420) -> dict:
    """Recessive-chrome plotly layout shared by all QuantLab figures."""
    return dict(
        title=dict(text=title, font=dict(size=16, color=PRIMARY_INK)),
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=60, b=50),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=PRIMARY_INK),
        xaxis=dict(title=x_title, gridcolor=GRID_COLOR, linecolor=AXIS_COLOR, tickfont=dict(color=MUTED_INK)),
        yaxis=dict(title=y_title, gridcolor=GRID_COLOR, linecolor=AXIS_COLOR, tickfont=dict(color=MUTED_INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        hovermode="x unified",
    )


def plot_date(x) -> str:
    """ISO string for shape/annotation coordinates — pd.Timestamp objects
    break kaleido's JSON serializer during static (PNG) export."""
    return pd.Timestamp(x).isoformat()


def shade_out_of_sample(fig: go.Figure, split_date, end_date) -> None:
    """Wash the out-of-sample region and label both segments."""
    split, end = plot_date(split_date), plot_date(end_date)
    fig.add_vrect(x0=split, x1=end, fillcolor=OOS_WASH, line_width=0, layer="below")
    fig.add_vline(x=split, line_dash="dot", line_color=MUTED_INK, line_width=1)
    fig.add_annotation(
        x=split, y=1.06, yref="paper", xanchor="right", showarrow=False,
        text="◀ in-sample", font=dict(size=11, color=MUTED_INK),
    )
    fig.add_annotation(
        x=split, y=1.06, yref="paper", xanchor="left", showarrow=False,
        text=" out-of-sample ▶", font=dict(size=11, color=MUTED_INK),
    )
