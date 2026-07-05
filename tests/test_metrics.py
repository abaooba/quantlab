"""Metric tests: every formula checked against a hand computation."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    TRADING_DAYS_PER_YEAR,
    annualized_volatility,
    cagr,
    drawdown_series,
    exposure,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    total_return,
    win_rate,
)


def eq(values, dates=None):
    if dates is None:
        return pd.Series(values, dtype=float)
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)


class TestCagr:
    def test_doubling_in_two_calendar_years(self):
        equity = eq([100, 150, 200], ["2020-01-01", "2021-01-01", "2021-12-31"])
        # 730 days ≈ 1.999 years; CAGR ≈ sqrt(2) - 1
        assert cagr(equity) == pytest.approx(np.sqrt(2) - 1, rel=2e-3)

    def test_range_index_uses_252(self):
        equity = pd.Series(np.linspace(100, 200, TRADING_DAYS_PER_YEAR + 1))
        assert cagr(equity) == pytest.approx(1.0, rel=1e-9)  # doubled in one trading year

    def test_wiped_out_account(self):
        equity = eq([100, 50, 0], ["2020-01-01", "2020-06-01", "2021-01-01"])
        assert cagr(equity) == -1.0

    def test_too_short(self):
        assert np.isnan(cagr(eq([100])))


class TestSharpe:
    def test_hand_computed(self):
        rets = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
        expected = rets.mean() / rets.std(ddof=1) * np.sqrt(252)
        assert sharpe_ratio(rets) == pytest.approx(expected)

    def test_risk_free_rate_reduces_sharpe(self):
        rets = pd.Series(np.full(252, 0.0005))  # ~13.4% ann., zero vol… almost
        rets.iloc[::2] += 0.0001  # give it a whisper of variance
        assert sharpe_ratio(rets, rf=0.05) < sharpe_ratio(rets, rf=0.0)

    def test_zero_variance_is_nan(self):
        assert np.isnan(sharpe_ratio(pd.Series([0.01] * 100)))

    def test_annualization_factor(self):
        # mean 4bp, sd 1% daily → Sharpe = .0004/.01*√252 ≈ 0.635
        rng = np.random.default_rng(0)
        rets = pd.Series(rng.normal(0.0004, 0.01, 100_000))
        assert sharpe_ratio(rets) == pytest.approx(0.635, abs=0.08)


class TestSortino:
    def test_hand_computed(self):
        rets = pd.Series([0.02, -0.01, 0.0, 0.01])
        downside_dev = np.sqrt(np.mean(np.minimum(rets, 0) ** 2))
        expected = rets.mean() / downside_dev * np.sqrt(252)
        assert sortino_ratio(rets) == pytest.approx(expected)

    def test_no_down_days_is_infinite(self):
        assert sortino_ratio(pd.Series([0.01, 0.02, 0.0, 0.03])) == np.inf

    def test_sqrt2_relationship_on_symmetric_returns(self):
        # For symmetric returns, downside deviation ≈ total std / √2, so
        # Sortino ≈ √2 × Sharpe — a classic identity worth pinning down.
        rng = np.random.default_rng(1)
        rets = pd.Series(rng.normal(0.0005, 0.01, 100_000))
        assert sortino_ratio(rets) / sharpe_ratio(rets) == pytest.approx(np.sqrt(2), rel=0.05)

    def test_downside_concentration_lowers_sortino_relative_to_sharpe(self):
        # Two streams, same Sharpe ingredients, but one hides its variance in
        # crashes: its Sortino/Sharpe ratio must be worse than the symmetric one.
        rng = np.random.default_rng(2)
        symmetric = pd.Series(rng.normal(0.0005, 0.01, 50_000))
        crashy = pd.Series(np.where(rng.random(50_000) < 0.03, rng.normal(-0.03, 0.01, 50_000), 0.002))
        sym_ratio = sortino_ratio(symmetric) / sharpe_ratio(symmetric)
        crashy_ratio = sortino_ratio(crashy) / sharpe_ratio(crashy)
        assert crashy_ratio < sym_ratio


class TestDrawdown:
    def test_known_path(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        equity = eq([100, 120, 90, 95, 130], dates)
        dd = max_drawdown(equity)
        assert dd.depth == pytest.approx(-0.25)  # 120 → 90
        assert dd.peak_date == dates[1]
        assert dd.trough_date == dates[2]
        assert dd.recovery_date == dates[4]

    def test_never_recovered(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        dd = max_drawdown(eq([100, 120, 80, 90], dates))
        assert dd.depth == pytest.approx(-1 / 3)
        assert dd.recovery_date is None

    def test_monotonic_curve_has_zero_drawdown(self):
        dd = max_drawdown(eq([100, 110, 120, 130]))
        assert dd.depth == 0.0
        assert dd.peak_date is None

    def test_drawdown_series_shape(self):
        dds = drawdown_series(eq([100, 120, 90]))
        assert dds.tolist() == pytest.approx([0.0, 0.0, -0.25])


class TestRates:
    def test_win_rate_ignores_flat_days(self):
        rets = pd.Series([0.01, -0.01, 0.0, 0.02, 0.0])
        assert win_rate(rets) == pytest.approx(2 / 3)

    def test_win_rate_all_flat_is_nan(self):
        assert np.isnan(win_rate(pd.Series([0.0, 0.0])))

    def test_exposure(self):
        assert exposure(pd.Series([0, 1, 1, 0, -1, 0])) == pytest.approx(0.5)

    def test_total_return(self):
        assert total_return(eq([100, 150])) == pytest.approx(0.5)

    def test_volatility_annualization(self):
        rets = pd.Series([0.01, -0.01] * 50)
        assert annualized_volatility(rets) == pytest.approx(rets.std(ddof=1) * np.sqrt(252))


class TestSummarize:
    def test_all_keys_present(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        rets = pd.Series([0.0, 0.01, -0.005, 0.02, 0.0, 0.01, -0.01, 0.0, 0.005, 0.01], index=idx)
        results = pd.DataFrame(
            {
                "daily_return": rets,
                "equity": 100_000 * (1 + rets).cumprod(),
                "position": (rets != 0).astype(float),
            }
        )
        out = summarize(results)
        assert set(out) == {
            "Total return", "CAGR", "Volatility (ann.)", "Sharpe", "Sortino",
            "Max drawdown", "Win rate (daily)", "Exposure",
        }
