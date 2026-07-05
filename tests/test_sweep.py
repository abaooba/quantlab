"""Parameter-sweep tests: grids, ranks, and the heatmap pair."""

import numpy as np
import pandas as pd
import pytest

from src.sweep import best_in_sample, oos_rank_of_is_best, parameter_sweep, sweep_heatmap_pair


def frame(n=400, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-01", periods=n)
    closes = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.011, n)), index=idx)
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1e6})


class TestSweep:
    def test_valid_combos_only(self):
        # 3×3 grid, but combos with fast ≥ slow are invalid for MA crossover.
        grid = {"fast": [5, 20, 60], "slow": [30, 50, 10]}
        df = parameter_sweep("MA Crossover", frame(), grid)
        valid = [(f, s) for f in grid["fast"] for s in grid["slow"] if f < s]
        assert len(df) == len(valid)
        assert {"is_sharpe", "oos_sharpe", "is_cagr", "oos_cagr"} <= set(df.columns)

    def test_all_invalid_raises(self):
        with pytest.raises(ValueError, match="no valid"):
            parameter_sweep("MA Crossover", frame(), {"fast": [50], "slow": [20]})

    def test_oversized_grid_raises(self):
        big = {"fast": list(range(2, 40)), "slow": list(range(41, 80))}
        with pytest.raises(ValueError, match="thin the grid"):
            parameter_sweep("MA Crossover", frame(), big)

    def test_best_and_rank(self):
        df = parameter_sweep("MA Crossover", frame(), {"fast": [5, 10, 20], "slow": [30, 60]})
        best = best_in_sample(df)
        assert best["is_sharpe"] == df["is_sharpe"].max()
        rank, pct = oos_rank_of_is_best(df)
        assert 1 <= rank <= len(df)
        assert 0.0 <= pct <= 1.0


class TestHeatmap:
    def test_pair_figure(self):
        df = parameter_sweep("MA Crossover", frame(), {"fast": [5, 10, 20], "slow": [30, 45, 60]})
        fig = sweep_heatmap_pair(df, x="fast", y="slow")
        heatmaps = [t for t in fig.data if t.type == "heatmap"]
        assert len(heatmaps) == 2
        # shared, zero-centered color scale
        assert heatmaps[0].zmin == -heatmaps[0].zmax
