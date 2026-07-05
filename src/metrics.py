"""Performance metrics — the numbers professionals actually judge a strategy by.

Conventions used throughout:
- ``daily_returns`` are simple (not log) daily returns of the *strategy*,
  net of costs; ``equity`` is the compounded dollar curve.
- Annualization uses 252 trading days. Sharpe/Sortino annualize by √252,
  which assumes independent daily returns (documented, imperfect, standard).
- ``rf`` is an *annual* risk-free rate (e.g. 0.05 for 5%), de-annualized to
  daily by division — fine at these magnitudes.
- Metrics that are undefined on the given data (zero variance, empty input)
  return ``nan`` rather than raising: the UI renders them as "—".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Variance below this is numerical noise, not information: pandas can return
# ~1e-18 for the std of a literally constant series, which would otherwise
# manufacture Sharpe ratios in the quadrillions.
_EPS_STD = 1e-12


def total_return(equity: pd.Series) -> float:
    """End-to-end simple return of the equity curve."""
    if len(equity) < 2:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate from first to last equity value.

    Uses calendar time (days/365.25) when the index is datetime-like —
    matching how published CAGRs are quoted — and bar count / 252 otherwise.
    """
    if len(equity) < 2:
        return float("nan")
    if isinstance(equity.index, pd.DatetimeIndex):
        years = (equity.index[-1] - equity.index[0]).days / 365.25
    else:
        years = (len(equity) - 1) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return float("nan")
    growth = equity.iloc[-1] / equity.iloc[0]
    if growth <= 0:
        return -1.0  # account wiped out (or worse)
    return float(growth ** (1.0 / years) - 1.0)


def annualized_volatility(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2:
        return float("nan")
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sharpe: mean excess return per unit of total volatility.

    The single most-quoted risk-adjusted metric — and one whose assumptions
    (symmetric, i.i.d., normal-ish returns) deserve the skepticism the rest
    of this project applies to backtests.
    """
    if len(daily_returns) < 2:
        return float("nan")
    excess = daily_returns - rf / TRADING_DAYS_PER_YEAR
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < _EPS_STD:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    """Like Sharpe, but the denominator only counts *downside* deviation.

    Downside deviation is the root-mean-square of negative excess returns
    taken over **all** observations (the full-sample convention), so calm
    positive days still dilute the penalty — only losses add to it.
    Returns +inf for a strategy with positive mean and literally no down
    days (mathematically earned, practically a red flag).
    """
    if len(daily_returns) < 2:
        return float("nan")
    excess = daily_returns - rf / TRADING_DAYS_PER_YEAR
    downside = excess.clip(upper=0.0)
    dd = np.sqrt(float((downside**2).mean()))
    mean = float(excess.mean())
    if dd < _EPS_STD:
        return float("inf") if mean > 0 else float("nan")
    return float(mean / dd * np.sqrt(TRADING_DAYS_PER_YEAR))


@dataclass(frozen=True)
class DrawdownResult:
    """Deepest peak-to-trough decline and when it happened."""

    depth: float  # negative fraction, e.g. -0.34 = -34%
    peak_date: object  # index label of the high-water mark before the trough
    trough_date: object  # index label of the low point
    recovery_date: object | None  # first index label back at the peak, None if never


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fraction below the running high-water mark at every bar (≤ 0)."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> DrawdownResult:
    if len(equity) == 0:
        return DrawdownResult(float("nan"), None, None, None)
    dd = drawdown_series(equity)
    trough = dd.idxmin()
    depth = float(dd.loc[trough])
    if depth == 0.0:
        return DrawdownResult(0.0, None, None, None)  # curve never declined
    pre = equity.loc[:trough]
    peak = pre.idxmax()
    post = equity.loc[trough:]
    recovered = post[post >= equity.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    return DrawdownResult(depth, peak, trough, recovery)


def win_rate(daily_returns: pd.Series) -> float:
    """Fraction of *active* days that were profitable.

    Zero-return days (almost always days the strategy was flat) are
    excluded, so a strategy in the market 10% of the time isn't flattered
    or damned by the 90% it sat out. Per-trade win rate — usually the more
    meaningful number — lives in ``src.trades.trade_stats``.
    """
    active = daily_returns[daily_returns != 0.0]
    if len(active) == 0:
        return float("nan")
    return float((active > 0).mean())


def exposure(positions: pd.Series) -> float:
    """Fraction of bars holding any position — how often capital was at risk."""
    if len(positions) == 0:
        return float("nan")
    return float((positions != 0).mean())


def build_equity(
    daily_returns: pd.Series, initial_capital: float = 100_000.0, base_date=None
) -> pd.Series:
    """Compound daily returns into an equity curve.

    When ``base_date`` is given, a starting row equal to ``initial_capital``
    is prepended at that date. This matters for *segments*: the first bar of
    an out-of-sample slice usually has a nonzero return, and measuring
    growth from that bar's closing equity would silently drop it.
    """
    equity = initial_capital * (1.0 + daily_returns).cumprod()
    if base_date is not None:
        base = pd.Series([float(initial_capital)], index=pd.Index([base_date]))
        equity = pd.concat([base, equity])
    return equity


def summarize_returns(
    daily_returns: pd.Series,
    positions: pd.Series | None = None,
    initial_capital: float = 100_000.0,
    rf: float = 0.0,
    base_date=None,
) -> dict[str, float]:
    """All headline metrics for a daily-return stream, as a flat dict.

    ``Exposure`` is included only when ``positions`` is provided.
    """
    equity = build_equity(daily_returns, initial_capital, base_date)
    mdd = max_drawdown(equity)
    out = {
        "Total return": total_return(equity),
        "CAGR": cagr(equity),
        "Volatility (ann.)": annualized_volatility(daily_returns),
        "Sharpe": sharpe_ratio(daily_returns, rf),
        "Sortino": sortino_ratio(daily_returns, rf),
        "Max drawdown": mdd.depth,
        "Win rate (daily)": win_rate(daily_returns),
    }
    if positions is not None:
        out["Exposure"] = exposure(positions)
    return out


def summarize(results: pd.DataFrame, rf: float = 0.0) -> dict[str, float]:
    """Headline metrics for an engine result frame.

    The engine guarantees the first bar is flat (zero return), so the first
    equity value *is* the initial capital and no base row is needed.
    """
    return summarize_returns(
        results["daily_return"],
        positions=results["position"],
        initial_capital=float(results["equity"].iloc[0]),
        rf=rf,
    )


def alpha_beta(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, rf: float = 0.0
) -> dict[str, float]:
    """CAPM regression of strategy on benchmark excess daily returns.

    Fits ``r_s − rf = α + β·(r_b − rf) + ε`` by least squares and reports:
    ``beta`` (market sensitivity: 1 = moves with the market, 0 = market-
    neutral), ``alpha_ann`` (the daily intercept × 252 — return unexplained
    by market exposure, the number active managers are paid for), and ``r2``
    (how much of the strategy's variance the market explains).
    """
    joined = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(joined) < 3:
        return {"alpha_ann": float("nan"), "beta": float("nan"), "r2": float("nan")}
    rf_d = rf / TRADING_DAYS_PER_YEAR
    y = joined.iloc[:, 0].to_numpy() - rf_d
    x = joined.iloc[:, 1].to_numpy() - rf_d
    var_x = x.var()
    if var_x < _EPS_STD**2:
        return {"alpha_ann": float("nan"), "beta": float("nan"), "r2": float("nan")}
    beta = float(np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1))
    alpha_d = float(y.mean() - beta * x.mean())
    resid = y - (alpha_d + beta * x)
    var_y = y.var(ddof=1)
    r2 = float(1.0 - resid.var(ddof=1) / var_y) if var_y > _EPS_STD**2 else float("nan")
    return {"alpha_ann": alpha_d * TRADING_DAYS_PER_YEAR, "beta": beta, "r2": r2}


def information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Mean active return over tracking error, annualized.

    Sharpe grades a strategy against cash; the information ratio grades it
    against the benchmark it is trying to beat — the metric that matters
    once "just buy the index" is on the table.
    """
    active = (strategy_returns - benchmark_returns).dropna()
    if len(active) < 2:
        return float("nan")
    te = active.std(ddof=1)
    if not np.isfinite(te) or te < _EPS_STD:
        return float("nan")
    return float(active.mean() / te * np.sqrt(TRADING_DAYS_PER_YEAR))


def rolling_sharpe(daily_returns: pd.Series, window: int = 252, rf: float = 0.0) -> pd.Series:
    """Trailing-window annualized Sharpe — how the 'one number' drifts.

    A full-period Sharpe hides regime dependence; the rolling view shows a
    strategy living and dying with market conditions.
    """
    excess = daily_returns - rf / TRADING_DAYS_PER_YEAR
    mean = excess.rolling(window).mean()
    sd = excess.rolling(window).std(ddof=1)
    out = mean / sd * np.sqrt(TRADING_DAYS_PER_YEAR)
    return out.where(sd >= _EPS_STD)
