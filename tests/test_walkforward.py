"""Walk-forward tests: window mechanics and out-of-sample chaining."""

import numpy as np
import pandas as pd
import pytest

from src.walkforward import plot_walk_forward, walk_forward


def frame(n=600, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n)
    closes = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n)), index=idx)
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1e6})


GRID = {"fast": [5, 10], "slow": [30, 60]}


class TestMechanics:
    def test_windows_tile_the_test_region(self):
        prices = frame()
        result = walk_forward("MA Crossover", prices, GRID, n_windows=4, initial_train_frac=0.4)
        assert len(result.windows) == 4
        first_test = int(len(prices) * 0.4)
        assert result.windows[0].test_start == prices.index[first_test]
        assert result.windows[-1].test_end == prices.index[-1]
        for prev, nxt in zip(result.windows, result.windows[1:]):
            assert prev.test_end < nxt.test_start  # no overlap, no gap in bars

    def test_oos_returns_cover_exactly_the_test_region(self):
        prices = frame()
        result = walk_forward("MA Crossover", prices, GRID, n_windows=4, initial_train_frac=0.4)
        first_test = int(len(prices) * 0.4)
        assert result.oos_returns.index.equals(prices.index[first_test:])

    def test_anchored_training_expands(self):
        prices = frame()
        result = walk_forward("MA Crossover", prices, GRID, n_windows=3, initial_train_frac=0.5)
        ends = [w.train_end for w in result.windows]
        assert ends == sorted(ends)
        assert all(w.train_start == prices.index[0] for w in result.windows)
        for w in result.windows:
            assert w.train_end < w.test_start

    def test_chosen_params_come_from_grid(self):
        result = walk_forward("MA Crossover", frame(), GRID, n_windows=3)
        for w in result.windows:
            assert w.best_params["fast"] in GRID["fast"]
            assert w.best_params["slow"] in GRID["slow"]

    def test_invalid_combos_skipped(self):
        # fast ≥ slow combos must be dropped, not crash the run.
        grid = {"fast": [10, 60], "slow": [30]}
        result = walk_forward("MA Crossover", frame(), grid, n_windows=3)
        assert all(w.best_params == {"fast": 10, "slow": 30} for w in result.windows)

    def test_all_invalid_grid_raises(self):
        with pytest.raises(ValueError, match="no valid parameter"):
            walk_forward("MA Crossover", frame(), {"fast": [50], "slow": [20]}, n_windows=3)

    def test_too_short_series_raises(self):
        with pytest.raises(ValueError, match="too short"):
            walk_forward("MA Crossover", frame(40), GRID, n_windows=5)


class TestResults:
    def test_metrics_and_hindsight_reported(self):
        result = walk_forward("MA Crossover", frame(), GRID, n_windows=4)
        assert "Sharpe" in result.oos_metrics
        assert "CAGR" in result.benchmark_metrics
        assert result.hindsight_params["fast"] in GRID["fast"]
        assert np.isfinite(result.hindsight_sharpe)

    def test_oos_equity_compounds_oos_returns_from_a_base_row(self):
        prices = frame()
        result = walk_forward("MA Crossover", prices, GRID, n_windows=4)
        first_test = int(len(prices) * 0.4)
        # curve is anchored at initial capital on the last train-only bar…
        assert result.oos_equity.index[0] == prices.index[first_test - 1]
        assert result.oos_equity.iloc[0] == 100_000
        # …and then compounds exactly the chained out-of-sample returns
        expected = 100_000 * (1 + result.oos_returns).cumprod()
        pd.testing.assert_series_equal(
            result.oos_equity.iloc[1:], expected, check_freq=False, check_names=False
        )

    def test_plot_builds(self):
        result = walk_forward("MA Crossover", frame(), GRID, n_windows=3)
        fig = plot_walk_forward(result)
        assert len(fig.data) >= 1
