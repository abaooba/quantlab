"""Regime-attribution tests: classification thresholds and per-regime stats."""

import numpy as np
import pandas as pd
import pytest

from src.regimes import classify_vix, regime_performance, vix_regimes


def idx(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


class TestClassify:
    def test_thresholds(self):
        vix = pd.Series([12.0, 15.0, 20.0, 25.0, 30.0, np.nan], index=idx(6))
        labels = classify_vix(vix)
        assert labels.tolist()[:5] == ["calm", "normal", "normal", "normal", "stressed"]
        assert pd.isna(labels.iloc[5])

    def test_boundaries_are_exclusive(self):
        vix = pd.Series([14.999, 25.001], index=idx(2))
        assert classify_vix(vix).tolist() == ["calm", "stressed"]

    def test_bad_thresholds_raise(self):
        with pytest.raises(ValueError):
            classify_vix(pd.Series([20.0], index=idx(1)), calm_below=30, stressed_above=20)


class TestPerformance:
    def test_per_regime_stats_hand_checked(self):
        n = 9
        regimes = pd.Series(["calm"] * 3 + ["normal"] * 3 + ["stressed"] * 3, index=idx(n))
        # strategy earns only in calm; loses in stressed
        strat = pd.Series([0.01, 0.01, 0.01, 0.0, 0.0, 0.0, -0.02, -0.02, -0.02], index=idx(n))
        bench = pd.Series(0.001, index=idx(n))
        table = regime_performance({"Strat": strat, "B&H": bench}, regimes)

        assert table.loc["calm", "Days"] == 3
        assert table.loc["calm", "Strat · ann. return"] == pytest.approx(0.01 * 252)
        assert table.loc["stressed", "Strat · ann. return"] == pytest.approx(-0.02 * 252)
        # zero-variance segments give NaN Sharpe rather than nonsense
        assert np.isnan(table.loc["normal", "Strat · Sharpe"])

    def test_all_three_regimes_always_present(self):
        regimes = pd.Series(["calm"] * 5, index=idx(5))
        rets = pd.Series(0.01, index=idx(5))
        table = regime_performance({"S": rets}, regimes)
        assert list(table.index) == ["calm", "normal", "stressed"]
        assert table.loc["stressed", "Days"] == 0


@pytest.mark.network
class TestLiveVIX:
    def test_vix_regimes_align_to_prices(self, tmp_path):
        from src.data import fetch_prices

        prices = fetch_prices("SPY", "2020-01-01", "2020-12-31", cache_dir=tmp_path)
        regimes = vix_regimes(prices, cache_dir=tmp_path)
        assert regimes is not None
        assert regimes.index.equals(prices.index)
        # March 2020: if any period in history was 'stressed', it's this one
        assert (regimes.loc["2020-03"] == "stressed").all()
