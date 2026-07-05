"""Extension tests: CAPM regression, information ratio, rolling Sharpe,
volatility targeting, and the expected-max-Sharpe luck yardstick."""

import numpy as np
import pandas as pd
import pytest

from src.engine import run_backtest
from src.metrics import (
    alpha_beta,
    annualized_volatility,
    information_ratio,
    rolling_sharpe,
    sharpe_ratio,
)
from src.sizing import volatility_target
from src.stats import expected_max_sharpe


def bench_returns(n=5000, seed=4, mean=0.0003, sd=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(rng.normal(mean, sd, n), index=idx)


class TestAlphaBeta:
    def test_recovers_constructed_beta_and_alpha(self):
        rb = bench_returns()
        alpha_daily = 0.0002
        rs = alpha_daily + 0.5 * rb  # exact linear relationship
        out = alpha_beta(rs, rb)
        assert out["beta"] == pytest.approx(0.5, abs=1e-9)
        assert out["alpha_ann"] == pytest.approx(alpha_daily * 252, rel=1e-6)
        assert out["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_noise_lowers_r2_not_beta_much(self):
        rng = np.random.default_rng(5)
        rb = bench_returns()
        rs = 0.8 * rb + pd.Series(rng.normal(0, 0.005, len(rb)), index=rb.index)
        out = alpha_beta(rs, rb)
        assert out["beta"] == pytest.approx(0.8, abs=0.05)
        assert 0.4 < out["r2"] < 0.9

    def test_benchmark_against_itself(self):
        rb = bench_returns()
        out = alpha_beta(rb, rb)
        assert out["beta"] == pytest.approx(1.0)
        assert out["alpha_ann"] == pytest.approx(0.0, abs=1e-12)
        assert out["r2"] == pytest.approx(1.0)

    def test_flat_benchmark_is_nan(self):
        rb = pd.Series(0.0, index=pd.bdate_range("2020-01-01", periods=100))
        rs = bench_returns(100)
        rs.index = rb.index
        out = alpha_beta(rs, rb)
        assert np.isnan(out["beta"])


class TestInformationRatio:
    def test_hand_computed(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        rs = pd.Series([0.02, 0.00, 0.01, -0.01], index=idx)
        rb = pd.Series([0.01, 0.01, 0.00, 0.00], index=idx)
        active = rs - rb
        expected = active.mean() / active.std(ddof=1) * np.sqrt(252)
        assert information_ratio(rs, rb) == pytest.approx(expected)

    def test_identical_series_is_nan(self):
        rb = bench_returns(200)
        assert np.isnan(information_ratio(rb, rb))


class TestRollingSharpe:
    def test_window_value_matches_static_sharpe(self):
        rets = bench_returns(600)
        roll = rolling_sharpe(rets, window=252)
        # the value at bar i equals the plain Sharpe of the trailing window
        i = 400
        window = rets.iloc[i - 251 : i + 1]
        assert roll.iloc[i] == pytest.approx(sharpe_ratio(window))

    def test_warmup_is_nan_and_index_preserved(self):
        rets = bench_returns(300)
        roll = rolling_sharpe(rets, window=252)
        assert roll.index.equals(rets.index)
        assert roll.iloc[:251].isna().all()


class TestVolatilityTargeting:
    def heteroskedastic_prices(self, n=1000, seed=6):
        rng = np.random.default_rng(seed)
        vols = np.where(np.arange(n) < n // 2, 0.004, 0.030)  # calm half, wild half
        rets = rng.normal(0.0003, 1.0, n) * vols
        closes = 100 * np.cumprod(1 + rets)
        idx = pd.bdate_range("2015-01-01", periods=n)
        s = pd.Series(closes, index=idx)
        return pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s, "Volume": 1e6})

    def test_sizing_pulls_realized_vol_toward_target(self):
        prices = self.heteroskedastic_prices()
        always = pd.Series(1.0, index=prices.index)
        target = 0.08

        raw = run_backtest(prices, always, cost_bps=0)
        sized = run_backtest(prices, volatility_target(always, prices, target_vol=target), cost_bps=0)

        # in the wild second half, sized vol should sit far closer to target
        half = prices.index[len(prices) // 2 + 30 :]
        raw_vol = annualized_volatility(raw.loc[half, "daily_return"])
        sized_vol = annualized_volatility(sized.loc[half, "daily_return"])
        assert abs(sized_vol - target) < abs(raw_vol - target)
        assert sized_vol == pytest.approx(target, rel=0.5)

    def test_bounds_and_warmup(self):
        prices = self.heteroskedastic_prices()
        always = pd.Series(1.0, index=prices.index)
        sized = volatility_target(always, prices, target_vol=0.10, lookback=20)
        assert (sized >= 0).all() and (sized <= 1.0).all()
        assert (sized.iloc[:20] == 0.0).all()  # no vol estimate yet → no trade

    def test_flat_prices_never_trade(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        s = pd.Series(100.0, index=idx)
        prices = pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s, "Volume": 1e6})
        sized = volatility_target(pd.Series(1.0, index=idx), prices)
        assert (sized == 0.0).all()

    def test_causality_scale_unchanged_by_future(self):
        prices = self.heteroskedastic_prices()
        always = pd.Series(1.0, index=prices.index)
        full = volatility_target(always, prices)
        t = 500
        truncated = volatility_target(always.iloc[: t + 1], prices.iloc[: t + 1])
        assert full.iloc[t] == pytest.approx(truncated.iloc[t])

    def test_validation(self):
        prices = self.heteroskedastic_prices(100)
        sig = pd.Series(1.0, index=prices.index)
        with pytest.raises(ValueError):
            volatility_target(sig, prices, target_vol=0.0)
        with pytest.raises(ValueError):
            volatility_target(sig, prices, max_leverage=2.0)


class TestExpectedMaxSharpe:
    def test_case_study_value(self):
        # 58 combos on the 1,761-bar SPY in-sample window → ≈ 0.88
        assert expected_max_sharpe(58, 1761) == pytest.approx(0.88, abs=0.01)

    def test_monotone_in_trials(self):
        assert expected_max_sharpe(2, 1000) < expected_max_sharpe(50, 1000) < expected_max_sharpe(1000, 1000)

    def test_shrinks_with_more_data(self):
        assert expected_max_sharpe(58, 5000) < expected_max_sharpe(58, 500)

    def test_degenerate_inputs(self):
        assert np.isnan(expected_max_sharpe(1, 1000))
        assert np.isnan(expected_max_sharpe(58, 1))
