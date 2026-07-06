"""Statistical honesty tools: is that Sharpe ratio even real?

A backtest hands you ONE realization of history. A Sharpe of 0.9 measured
on ten years of daily returns is an *estimate* with sampling error, not a
fact — and daily returns are autocorrelated and volatility-clustered, so
naive i.i.d. error bars are too tight. The moving-block bootstrap resamples
contiguous *blocks* of returns (default ~1 trading month), preserving
short-range dependence, and rebuilds the Sharpe distribution: if the 95%
confidence interval straddles zero, the honest summary of the backtest is
"we cannot tell whether this strategy has an edge."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import TRADING_DAYS_PER_YEAR


def _block_sample(x: np.ndarray, length: int, n_paths: int, block: int, rng) -> np.ndarray:
    """(n_paths, length) matrix of moving-block bootstrap resamples of ``x``."""
    n = len(x)
    n_blocks = int(np.ceil(length / block))
    starts = rng.integers(0, n - block + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :length]
    return x[idx]


@dataclass(frozen=True)
class BootstrapResult:
    point: float  # Sharpe measured on the actual sample
    lo: float  # lower CI bound
    hi: float  # upper CI bound
    level: float  # e.g. 0.95
    p_leq_zero: float  # fraction of resamples with Sharpe ≤ 0
    n_boot: int
    block: int

    def straddles_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi


def block_bootstrap_sharpe(
    daily_returns: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    level: float = 0.95,
    rf: float = 0.0,
    seed: int = 0,
) -> BootstrapResult:
    """Moving-block bootstrap confidence interval for the annualized Sharpe.

    Blocks of ``block`` consecutive days are drawn with replacement and
    concatenated to the original length; each synthetic history yields one
    Sharpe. Percentiles of that distribution form the CI. ``block`` ≈ 21
    (one trading month) is long enough to keep volatility clustering,
    short enough to still shuffle regimes.
    """
    x = daily_returns.to_numpy(dtype=float) - rf / TRADING_DAYS_PER_YEAR
    n = len(x)
    if n < 60:
        raise ValueError(f"need ≥ 60 daily returns for a meaningful bootstrap, got {n}")
    block = int(min(block, max(5, n // 10)))

    sd = x.std(ddof=1)
    point = float("nan") if sd < 1e-12 else float(x.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))

    rng = np.random.default_rng(seed)
    samples = _block_sample(x, n, n_boot, block, rng)  # (n_boot, n)

    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(sds > 1e-12, means / sds * np.sqrt(TRADING_DAYS_PER_YEAR), np.nan)
    sharpes = sharpes[np.isfinite(sharpes)]
    if len(sharpes) == 0:
        return BootstrapResult(point, float("nan"), float("nan"), level, float("nan"), n_boot, block)

    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(sharpes, [alpha, 1.0 - alpha])
    return BootstrapResult(
        point=point,
        lo=float(lo),
        hi=float(hi),
        level=level,
        p_leq_zero=float((sharpes <= 0.0).mean()),
        n_boot=n_boot,
        block=block,
    )


def expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """Expected best annualized Sharpe among ``n_trials`` ZERO-skill strategies.

    The selection-bias yardstick for parameter sweeps (Bailey & López de
    Prado's "expected maximum Sharpe"): even if every combination in a grid
    is pure noise, the *best* of N noisy Sharpe estimates is far above zero.
    Under H0 (no skill, roughly i.i.d. daily returns) an annualized Sharpe
    measured on ``n_obs`` daily bars has standard error ≈ √(252 / n_obs),
    and the expected maximum of N standard normals is approximately

        E[max] ≈ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)),   γ ≈ 0.5772

    Multiply the two and you get the in-sample Sharpe that luck *alone* was
    expected to hand the sweep's champion. An observed champion near or
    below this line is indistinguishable from noise. (Grid combos are
    positively correlated — they share one history — so the effective N is
    smaller and this line is, if anything, generous to the strategy.)
    """
    if n_trials < 2 or n_obs < 2:
        return float("nan")
    from statistics import NormalDist

    gamma = 0.5772156649015329  # Euler–Mascheroni
    ndist = NormalDist()
    z = (1 - gamma) * ndist.inv_cdf(1 - 1 / n_trials) + gamma * ndist.inv_cdf(
        1 - 1 / (n_trials * np.e)
    )
    se_annual = np.sqrt(TRADING_DAYS_PER_YEAR / n_obs)
    return float(se_annual * z)


@dataclass(frozen=True)
class DrawdownDistribution:
    """Bootstrap distribution of the max drawdown over a fixed horizon."""

    median: float  # typical worst peak-to-trough over the horizon (negative)
    p95: float  # 95th-percentile pain: only 5% of paths were worse
    horizon_years: float
    n_paths: int
    block: int


def bootstrap_drawdown_distribution(
    daily_returns: pd.Series,
    horizon_years: float = 3.0,
    n_paths: int = 2000,
    block: int = 21,
    seed: int = 0,
) -> DrawdownDistribution:
    """What max drawdown should you *expect* — even if the returns are real?

    ``max_drawdown`` on the backtest reports the one drawdown history
    happened to serve. But drawdown is a random variable: resample the daily
    returns (moving blocks, preserving volatility clustering) into thousands
    of alternate ``horizon_years``-long paths and record each path's worst
    peak-to-trough. The median answers "what's typical?"; the 95th
    percentile answers "what should I be *prepared* for before I'd have any
    statistical right to be surprised?" Traders who quit at the historical
    max drawdown usually quit inside this distribution's ordinary range.
    """
    x = daily_returns.to_numpy(dtype=float)
    n = len(x)
    if n < 60:
        raise ValueError(f"need ≥ 60 daily returns, got {n}")
    block = int(min(block, max(5, n // 10)))
    length = int(round(horizon_years * TRADING_DAYS_PER_YEAR))
    if length < 2:
        raise ValueError("horizon too short")

    rng = np.random.default_rng(seed)
    paths = _block_sample(x, length, n_paths, block, rng)
    equity = np.cumprod(1.0 + paths, axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    max_dd = (equity / running_max - 1.0).min(axis=1)

    return DrawdownDistribution(
        median=float(np.quantile(max_dd, 0.5)),
        p95=float(np.quantile(max_dd, 0.05)),
        horizon_years=horizon_years,
        n_paths=n_paths,
        block=block,
    )
