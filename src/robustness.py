"""Cross-asset robustness: does the edge travel, or does it live on one chart?

A rule tuned on SPY that only works on SPY isn't a strategy — it's a
description of SPY's past. Genuine effects (trend, mean reversion) are
supposed to be *pervasive*: they show up, weaker or stronger, across many
liquid instruments. Running the identical rule and parameters over a basket
of unrelated ETFs is therefore one of the cheapest and most damning
robustness tests available — and one almost no "amazing backtest" ever shows.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from src.data import fetch_prices
from src.evaluate import evaluate_strategy

# Liquid, long-history ETFs across distinct asset classes / geographies.
DEFAULT_BASKET = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT"]


def cross_asset_check(
    strategy: str | Callable[..., pd.Series],
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    train_frac: float = 0.7,
    cost_bps: float = 5.0,
    rf: float = 0.0,
    cache_dir=None,
    **params,
) -> pd.DataFrame:
    """Run one strategy, fixed parameters, across a basket of tickers.

    Returns a frame with one row per ticker that had data: in/out-of-sample
    Sharpe, out-of-sample CAGR, the buy-and-hold out-of-sample Sharpe on the
    same instrument, the strategy-minus-benchmark Sharpe edge, and whether
    the overfitting flag fired. Tickers whose data can't be fetched are
    skipped (they appear in the frame with ``bars = 0``).
    """
    tickers = tickers or DEFAULT_BASKET
    rows = []
    for ticker in tickers:
        kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
        prices = fetch_prices(ticker, start, end, **kwargs)
        if prices is None or len(prices) < 120:
            rows.append({"ticker": ticker, "bars": 0})
            continue
        try:
            r = evaluate_strategy(strategy, prices, train_frac=train_frac,
                                  cost_bps=cost_bps, rf=rf, **params)
        except (ValueError, KeyError):
            rows.append({"ticker": ticker, "bars": len(prices)})
            continue
        rows.append(
            {
                "ticker": ticker,
                "bars": len(prices),
                "is_sharpe": r.in_sample["Sharpe"],
                "oos_sharpe": r.out_of_sample["Sharpe"],
                "oos_cagr": r.out_of_sample["CAGR"],
                "bh_oos_sharpe": r.benchmark_out_of_sample["Sharpe"],
                "oos_edge": r.out_of_sample["Sharpe"] - r.benchmark_out_of_sample["Sharpe"],
                "overfit_flag": r.overfit,
            }
        )
    return pd.DataFrame(rows)


def robustness_summary(check: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for a cross-asset check.

    ``beat_benchmark_frac`` is the fraction of testable tickers where the
    strategy's out-of-sample Sharpe beat buy-and-hold on the same instrument
    — the question that matters once "just hold it" is the alternative.
    """
    valid = check.dropna(subset=["oos_sharpe"]) if "oos_sharpe" in check else check.iloc[0:0]
    if len(valid) == 0:
        return {"tickers_tested": 0}
    return {
        "tickers_tested": int(len(valid)),
        "median_oos_sharpe": float(valid["oos_sharpe"].median()),
        "positive_oos_frac": float((valid["oos_sharpe"] > 0).mean()),
        "beat_benchmark_frac": float((valid["oos_edge"] > 0).mean()),
        "overfit_flag_frac": float(valid["overfit_flag"].mean()),
    }
