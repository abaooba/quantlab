"""Evaluation-layer tests: the split, the verdict, the tables, the figures."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    comparison_table,
    evaluate_strategy,
    format_metric,
    overfitting_verdict,
    plot_drawdown,
    plot_equity_curve,
    split_in_out_sample,
)


def frame(n=400, seed=3, drift=0.0004):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    closes = pd.Series(100 * np.cumprod(1 + rng.normal(drift, 0.01, n)), index=idx)
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1e6})


def always_long(prices, **_):
    return pd.Series(1.0, index=prices.index)


class TestSplit:
    def test_split_is_chronological_at_the_fraction(self):
        prices = frame(100)
        split = split_in_out_sample(prices, 0.7)
        assert split == prices.index[70]

    def test_bad_fraction_raises(self):
        with pytest.raises(ValueError):
            split_in_out_sample(frame(100), 1.2)

    def test_tiny_series_raises(self):
        with pytest.raises(ValueError, match="too few"):
            split_in_out_sample(frame(3), 0.5)


class TestVerdict:
    def test_severe_overfit(self):
        flag, text = overfitting_verdict({"Sharpe": 2.0}, {"Sharpe": -0.5})
        assert flag and "🚨" in text

    def test_possible_overfit(self):
        flag, text = overfitting_verdict({"Sharpe": 2.0}, {"Sharpe": 0.6})
        assert flag and "⚠️" in text

    def test_consistent(self):
        flag, text = overfitting_verdict({"Sharpe": 1.0}, {"Sharpe": 0.8})
        assert not flag and "✅" in text

    def test_no_edge_anywhere(self):
        flag, text = overfitting_verdict({"Sharpe": -0.2}, {"Sharpe": -0.4})
        assert not flag and "➖" in text

    def test_nan_is_inconclusive(self):
        flag, text = overfitting_verdict({"Sharpe": float("nan")}, {"Sharpe": 1.0})
        assert not flag and "Not enough data" in text


class TestEvaluateStrategy:
    def test_always_long_tracks_benchmark(self):
        prices = frame()
        result = evaluate_strategy(always_long, prices, train_frac=0.7, cost_bps=5)
        # Same trades as buy-and-hold → same metrics to the penny.
        assert result.in_sample["CAGR"] == pytest.approx(result.benchmark_in_sample["CAGR"])
        assert result.out_of_sample["Sharpe"] == pytest.approx(result.benchmark_out_of_sample["Sharpe"])

    def test_registry_name_lookup(self):
        prices = frame()
        result = evaluate_strategy("MA Crossover", prices, fast=10, slow=30)
        assert result.strategy_name == "MA Crossover"
        assert result.params == {"fast": 10, "slow": 30}

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="unknown strategy"):
            evaluate_strategy("Alpha Machine 9000", frame())

    def test_segments_are_disjoint_and_exhaustive(self):
        prices = frame(200)
        result = evaluate_strategy(always_long, prices, train_frac=0.6)
        n_is = (result.results.index < result.split_date).sum()
        assert n_is == 120
        assert len(result.results) == 200

    def test_segment_metrics_are_rebased(self):
        # OOS CAGR must reflect only OOS growth, not carry IS gains in.
        prices = frame()
        result = evaluate_strategy(always_long, prices)
        oos = result.results[result.results.index >= result.split_date]
        oos_growth = (1 + oos["daily_return"]).prod()
        assert result.out_of_sample["Total return"] == pytest.approx(oos_growth - 1)

    def test_verdict_always_present(self):
        result = evaluate_strategy(always_long, frame())
        assert result.verdict


class TestPresentation:
    def test_format_metric(self):
        assert format_metric("CAGR", 0.1234) == "+12.3%"
        assert format_metric("Sharpe", 1.234) == "1.23"
        assert format_metric("Sharpe", float("nan")) == "—"
        assert format_metric("Sortino", float("inf")) == "∞"
        assert format_metric("Max drawdown", -0.345) == "-34.5%"

    def test_comparison_table_shape(self):
        result = evaluate_strategy(always_long, frame())
        table = comparison_table(result)
        assert table.shape == (8, 4)
        assert "Strategy · out-of-sample" in table.columns

    def test_figures_have_strategy_and_benchmark(self):
        result = evaluate_strategy(always_long, frame())
        for fig in (plot_equity_curve(result), plot_drawdown(result)):
            assert len(fig.data) == 2  # benchmark + strategy


class TestReviewFixes:
    def test_inverse_surprise_verdict(self):
        flag, text = overfitting_verdict({"Sharpe": -0.5}, {"Sharpe": 0.8})
        assert not flag and "🍀" in text and "luck" in text.lower()

    def test_explicit_split_date_snaps_to_next_bar(self):
        prices = frame(100)
        target = prices.index[40] + pd.Timedelta(days=1)  # between bars
        split = split_in_out_sample(prices, split_date=target)
        assert split == prices.index[41]

    def test_split_date_flows_through_evaluate(self):
        prices = frame(100)
        result = evaluate_strategy(always_long, prices, split_date=prices.index[30])
        assert result.split_date == prices.index[30]

    def test_split_date_at_edge_raises(self):
        prices = frame(100)
        with pytest.raises(ValueError, match="fewer than 2 bars"):
            split_in_out_sample(prices, split_date=prices.index[-1])
