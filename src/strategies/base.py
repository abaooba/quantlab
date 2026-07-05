"""Strategy contract and registry.

A strategy is a plain function ``(prices: pd.DataFrame, **params) -> pd.Series``
returning a signal Series aligned to ``prices.index`` with values in
{-1, 0, 1} (short / flat / long). The value at date *t* is the stance decided
at *t*'s close — the engine, not the strategy, applies the one-bar execution
delay. Strategies must be **causal**: the signal at *t* may only use rows up
to and including *t* (rolling windows, EWMs — anything backward-looking).

``STRATEGY_REGISTRY`` maps display name → signal function (the minimal
contract the engine needs). ``STRATEGY_SPECS`` carries UI metadata (parameter
ranges, defaults, descriptions) so the Streamlit app can render controls for
any registered strategy without hardcoding imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter, with enough metadata to render a UI control."""

    name: str  # kwarg name passed to the signal function
    label: str  # human-readable label
    min: float
    max: float
    default: float
    step: float = 1
    kind: str = "int"  # "int" | "float" | "bool"
    help: str = ""


@dataclass(frozen=True)
class StrategySpec:
    name: str
    fn: Callable[..., pd.Series]
    params: tuple[ParamSpec, ...] = field(default_factory=tuple)
    description: str = ""


STRATEGY_REGISTRY: dict[str, Callable[..., pd.Series]] = {}
STRATEGY_SPECS: dict[str, StrategySpec] = {}


def register_strategy(name: str, params: tuple[ParamSpec, ...] = (), description: str = ""):
    """Decorator: register a signal function under a display name."""

    def decorator(fn: Callable[..., pd.Series]) -> Callable[..., pd.Series]:
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"strategy {name!r} already registered")
        STRATEGY_REGISTRY[name] = fn
        STRATEGY_SPECS[name] = StrategySpec(name=name, fn=fn, params=tuple(params), description=description)
        return fn

    return decorator


def close_series(prices: pd.DataFrame | pd.Series) -> pd.Series:
    """Extract the close column strategies compute their indicators on."""
    if isinstance(prices, pd.Series):
        return prices.astype(float)
    if "Close" not in prices.columns:
        raise ValueError("prices DataFrame must have a 'Close' column")
    return prices["Close"].astype(float)


def stateful_signal(
    index: pd.Index, entries: pd.Series, exits: pd.Series, value: float = 1.0
) -> pd.Series:
    """Build a hold-until-exit signal from entry/exit event series.

    ``entries``/``exits`` are boolean Series. An entry switches the signal to
    ``value``; an exit switches it to 0; between events the last state is
    held. When both fire on the same bar the exit wins (conservative).
    """
    state = pd.Series(float("nan"), index=index)
    state[entries.fillna(False).astype(bool)] = value
    state[exits.fillna(False).astype(bool)] = 0.0
    return state.ffill().fillna(0.0)
