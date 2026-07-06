"""Bootstrap tests: reproducibility and statistical sanity."""

import numpy as np
import pandas as pd
import pytest

from src.stats import block_bootstrap_sharpe


def returns(mean, sd, n=1500, seed=2):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, sd, n))


class TestBootstrap:
    def test_deterministic_given_seed(self):
        rets = returns(0.0005, 0.01)
        a = block_bootstrap_sharpe(rets, n_boot=500, seed=42)
        b = block_bootstrap_sharpe(rets, n_boot=500, seed=42)
        assert (a.lo, a.hi, a.p_leq_zero) == (b.lo, b.hi, b.p_leq_zero)

    def test_interval_brackets_point_estimate(self):
        rets = returns(0.0008, 0.01)
        res = block_bootstrap_sharpe(rets, n_boot=1000)
        assert res.lo < res.point < res.hi
        assert res.lo < res.hi

    def test_zero_edge_straddles_zero(self):
        # Demean exactly: the bootstrap CI brackets the SAMPLE Sharpe, so an
        # unlucky draw with nonzero sample mean would not straddle zero.
        rets = returns(0.0, 0.01)
        rets = rets - rets.mean()
        res = block_bootstrap_sharpe(rets, n_boot=1000)
        assert res.straddles_zero()
        assert 0.1 < res.p_leq_zero < 0.9  # roughly a coin flip, as it should be

    def test_strong_edge_probably_positive(self):
        rets = returns(0.003, 0.005)  # implausibly good strategy
        res = block_bootstrap_sharpe(rets, n_boot=1000)
        assert not res.straddles_zero()
        assert res.p_leq_zero < 0.01

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="≥ 60"):
            block_bootstrap_sharpe(pd.Series([0.01] * 30))

    def test_block_shrinks_for_smallish_samples(self):
        rets = returns(0.001, 0.01, n=80)
        res = block_bootstrap_sharpe(rets, n_boot=200, block=21)
        assert res.block == 8  # min(21, 80 // 10)


class TestDrawdownDistribution:
    def test_deterministic_and_ordered(self):
        from src.stats import bootstrap_drawdown_distribution

        rets = returns(0.0004, 0.01)
        a = bootstrap_drawdown_distribution(rets, n_paths=500, seed=3)
        b = bootstrap_drawdown_distribution(rets, n_paths=500, seed=3)
        assert (a.median, a.p95) == (b.median, b.p95)
        # p95 is the *worse* tail: more negative than the median
        assert a.p95 < a.median < 0.0

    def test_longer_horizon_deepens_expected_drawdown(self):
        from src.stats import bootstrap_drawdown_distribution

        rets = returns(0.0, 0.01)
        short = bootstrap_drawdown_distribution(rets, horizon_years=1.0, n_paths=500)
        long = bootstrap_drawdown_distribution(rets, horizon_years=5.0, n_paths=500)
        assert long.median < short.median

    def test_higher_vol_deepens_drawdown(self):
        from src.stats import bootstrap_drawdown_distribution

        calm = bootstrap_drawdown_distribution(returns(0.0003, 0.005), n_paths=500)
        wild = bootstrap_drawdown_distribution(returns(0.0003, 0.02, seed=5), n_paths=500)
        assert wild.median < calm.median

    def test_short_series_raises(self):
        from src.stats import bootstrap_drawdown_distribution

        with pytest.raises(ValueError, match="≥ 60"):
            bootstrap_drawdown_distribution(pd.Series([0.01] * 30))
