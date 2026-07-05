"""Round-trip trade ledger — turning a position series back into trades.

Daily win rate treats every calendar day as an event; traders think in
*trades*. This module reconstructs each round trip (entry → exit of a
constant nonzero stance) from a backtest's position column, so you can ask
the questions that actually diagnose a strategy: how many trades, how long
held, average win vs. average loss, profit factor.

Execution convention mirrors the engine: the position held during bar *t*
was put on at bar *t-1*'s close, so a stance spanning bars [e, x) trades in
at ``close[e-1]`` and out at ``close[x-1]``. Trade returns here are gross of
transaction costs (cost drag lives in the equity curve, where it can be
attributed unambiguously — a 1→-1 reversal's single fee spans two trades).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_trades(results: pd.DataFrame, prices: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """List every round-trip trade implied by ``results['position']``.

    Returns a DataFrame with one row per trade: ``direction`` (+1 long /
    -1 short), ``entry_date``, ``exit_date``, ``entry_price``, ``exit_price``,
    ``bars_held``, ``gross_return`` (compounded, sign-adjusted), and ``open``
    (True when the final trade is still on at the end of the series).
    """
    close = prices["Close"] if isinstance(prices, pd.DataFrame) else prices
    close = close.astype(float)
    pos = results["position"]
    asset_ret = results["asset_return"]

    rows = []
    current: dict | None = None

    values = pos.to_numpy()
    for t in range(len(values)):
        p = values[t]
        prev = values[t - 1] if t > 0 else 0.0
        if p == prev:
            continue
        if prev != 0.0 and current is not None:
            rows.append(_close_trade(current, t, pos.index, close, asset_ret))
            current = None
        if p != 0.0:
            # stance decided (and executed) at the close of bar t-1
            current = {"start": t, "size": p}

    if current is not None:
        rows.append(_close_trade(current, len(values), pos.index, close, asset_ret, still_open=True))

    columns = ["direction", "entry_date", "exit_date", "entry_price", "exit_price",
               "bars_held", "gross_return", "open"]
    return pd.DataFrame(rows, columns=columns)


def _close_trade(current, end, index, close, asset_ret, still_open=False) -> dict:
    e, size = current["start"], current["size"]
    # e == 0 means the frame starts mid-position (e.g. an out-of-sample slice):
    # price the entry at the first visible bar instead of wrapping to index -1.
    entry_i, exit_i = max(e - 1, 0), end - 1
    held = asset_ret.iloc[e:end]
    gross = float((1.0 + size * held).prod() - 1.0)
    return {
        "direction": int(np.sign(size)),
        "entry_date": index[entry_i],
        "exit_date": index[exit_i],
        "entry_price": float(close.iloc[entry_i]),
        "exit_price": float(close.iloc[exit_i]),
        "bars_held": end - e,
        "gross_return": gross,
        "open": still_open,
    }


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    """Headline per-trade statistics for a trade ledger."""
    if len(trades) == 0:
        return {"Trades": 0}
    r = trades["gross_return"]
    wins, losses = r[r > 0], r[r < 0]
    gross_profit, gross_loss = float(wins.sum()), float(-losses.sum())
    return {
        "Trades": int(len(trades)),
        "Win rate (per trade)": float((r > 0).mean()),
        "Avg win": float(wins.mean()) if len(wins) else float("nan"),
        "Avg loss": float(losses.mean()) if len(losses) else float("nan"),
        "Profit factor": (gross_profit / gross_loss) if gross_loss > 0
        else (float("inf") if gross_profit > 0 else float("nan")),
        "Avg bars held": float(trades["bars_held"].mean()),
        "Best trade": float(r.max()),
        "Worst trade": float(r.min()),
    }
