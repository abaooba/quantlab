"""Cross-asset robustness and ensemble tests (offline, injected data)."""

import numpy as np
import pandas as pd
import pytest

import src.robustness as robustness
from src.ensemble import combine_signals, ensemble_backtest, strategy_correlations
from src.robustness import cross_asset_check, robustness_summary


def synth_prices(seed, n=600, drift=0.0004):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    s = pd.Series(100 * np.cumprod(1 + rng.normal(drift, 0.012, n)), index=idx)
    return pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s, "Volume": 1e6})


class TestCrossAsset:
    @pytest.fixture
    def fake_fetch(self, monkeypatch):
        def fetch(ticker, start, end, **kwargs):
            if ticker == "DEAD":
                return None
            return synth_prices(seed=abs(hash(ticker)) % 1000)

        monkeypatch.setattr(robustness, "fetch_prices", fetch)

    def test_one_row_per_ticker_and_dead_tickers_kept(self, fake_fetch):
        df = cross_asset_check("MA Crossover", ["AAA", "BBB", "DEAD"], fast=10, slow=30)
        assert list(df["ticker"]) == ["AAA", "BBB", "DEAD"]
        assert df.loc[df["ticker"] == "DEAD", "bars"].iloc[0] == 0
        testable = df[df["bars"] > 0]
        assert {"is_sharpe", "oos_sharpe", "oos_edge", "overfit_flag"} <= set(testable.columns)
        assert testable["oos_sharpe"].notna().all()

    def test_edge_is_strategy_minus_benchmark(self, fake_fetch):
        df = cross_asset_check("MA Crossover", ["AAA"], fast=10, slow=30)
        row = df.iloc[0]
        assert row["oos_edge"] == pytest.approx(row["oos_sharpe"] - row["bh_oos_sharpe"])

    def test_summary(self, fake_fetch):
        df = cross_asset_check("MA Crossover", ["AAA", "BBB", "CCC", "DEAD"], fast=10, slow=30)
        s = robustness_summary(df)
        assert s["tickers_tested"] == 3
        assert 0.0 <= s["positive_oos_frac"] <= 1.0
        assert 0.0 <= s["beat_benchmark_frac"] <= 1.0

    def test_summary_of_nothing(self):
        assert robustness_summary(pd.DataFrame({"ticker": [], "bars": []}))["tickers_tested"] == 0


class TestCombineSignals:
    def test_equal_weight_average(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        a = pd.Series([1.0, 1.0, 0.0, 0.0], index=idx)
        b = pd.Series([1.0, 0.0, 0.0, -1.0], index=idx)
        combined = combine_signals({"a": a, "b": b})
        assert combined.tolist() == [1.0, 0.5, 0.0, -0.5]

    def test_custom_weights_normalized(self):
        idx = pd.bdate_range("2020-01-01", periods=2)
        a = pd.Series([1.0, 0.0], index=idx)
        b = pd.Series([0.0, 1.0], index=idx)
        combined = combine_signals({"a": a, "b": b}, weights={"a": 3.0, "b": 1.0})
        assert combined.tolist() == [0.75, 0.25]

    def test_empty_or_zero_weights_raise(self):
        with pytest.raises(ValueError):
            combine_signals({})
        idx = pd.bdate_range("2020-01-01", periods=2)
        with pytest.raises(ValueError):
            combine_signals({"a": pd.Series(1.0, index=idx)}, weights={"a": 0.0})


class TestEnsembleBacktest:
    def test_ensemble_runs_all_registered_strategies(self):
        prices = synth_prices(3)
        ens, indiv = ensemble_backtest(prices)
        assert set(indiv) == {"MA Crossover", "RSI Mean-Reversion", "Bollinger Breakout"}
        assert len(ens) == len(prices)
        # blended position is the average of the shifted individual stances
        stacked = sum(r["position"] for r in indiv.values()) / len(indiv)
        pd.testing.assert_series_equal(ens["position"], stacked, check_names=False)

    def test_ensemble_vol_at_most_max_individual(self):
        prices = synth_prices(9)
        ens, indiv = ensemble_backtest(prices, cost_bps=0)
        vols = {n: r["daily_return"].std() for n, r in indiv.items()}
        assert ens["daily_return"].std() <= max(vols.values()) + 1e-12


class TestCorrelations:
    def test_matrix_shape_and_symmetry(self):
        prices = synth_prices(5)
        _, indiv = ensemble_backtest(prices)
        corr = strategy_correlations(indiv)
        assert corr.shape == (3, 3)
        assert (corr.values.diagonal() == 1.0).all()
        assert corr.equals(corr.T)

    def test_shared_flat_days_excluded(self):
        idx = pd.bdate_range("2020-01-01", periods=8)
        # both flat for 4 days, then perfectly opposite for 4
        a = pd.DataFrame({"daily_return": [0, 0, 0, 0, 0.01, -0.01, 0.01, -0.01],
                          "position": 1.0}, index=idx)
        b = pd.DataFrame({"daily_return": [0, 0, 0, 0, -0.01, 0.01, -0.01, 0.01],
                          "position": 1.0}, index=idx)
        corr = strategy_correlations({"a": a, "b": b})
        # padding zeros would drag this toward 0; active-only shows the truth
        assert corr.loc["a", "b"] == pytest.approx(-1.0)


class TestReviewRound3Fixes:
    def test_all_dead_basket_keeps_stable_columns(self, monkeypatch):
        # When EVERY ticker fails, consumers still need the metric columns
        # to exist so dropna("oos_sharpe") empties the frame instead of
        # raising KeyError.
        monkeypatch.setattr(robustness, "fetch_prices", lambda *a, **k: None)
        df = cross_asset_check("MA Crossover", ["X", "Y"], fast=10, slow=30)
        assert "oos_sharpe" in df.columns
        assert df["oos_sharpe"].isna().all()
        assert len(df.dropna(subset=["oos_sharpe"])) == 0
        assert robustness_summary(df)["tickers_tested"] == 0

    def test_invalid_params_for_all_tickers_keeps_stable_columns(self, monkeypatch):
        monkeypatch.setattr(robustness, "fetch_prices",
                            lambda *a, **k: synth_prices(seed=1))
        # fast >= slow raises inside evaluate for every ticker
        df = cross_asset_check("MA Crossover", ["X", "Y"], fast=100, slow=50)
        assert "oos_sharpe" in df.columns
        assert df["oos_sharpe"].isna().all()

    def test_barely_overlapping_series_give_nan_not_spurious_one(self):
        rng = np.random.default_rng(0)
        ia = pd.bdate_range("2020-01-01", periods=10)
        ib = pd.bdate_range("2020-01-13", periods=10)  # 2-day overlap
        a = pd.DataFrame({"daily_return": rng.normal(0, 0.01, 10), "position": 1.0}, index=ia)
        b = pd.DataFrame({"daily_return": rng.normal(0, 0.01, 10), "position": 1.0}, index=ib)
        corr = strategy_correlations({"a": a, "b": b})
        assert np.isnan(corr.loc["a", "b"])  # 2 shared points is not a correlation
