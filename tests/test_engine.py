"""Engine tests: the honesty guarantees, proven on constructed data."""

import numpy as np
import pandas as pd
import pytest

from src.engine import run_backtest, run_naive_backtest_do_not_use

CAPITAL = 100_000.0


def make_prices(closes) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    closes = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1e6})


class TestBasics:
    def test_flat_signal_equity_never_moves(self):
        prices = make_prices([100, 105, 98, 110, 120])
        signals = pd.Series(0.0, index=prices.index)
        res = run_backtest(prices, signals, initial_capital=CAPITAL, cost_bps=5)
        assert (res["equity"] == CAPITAL).all()
        assert (res["cost"] == 0).all()

    def test_constant_long_matches_manual_buy_and_hold(self):
        closes = [100.0, 110.0, 121.0, 108.9]
        prices = make_prices(closes)
        signals = pd.Series(1.0, index=prices.index)
        res = run_backtest(prices, signals, initial_capital=CAPITAL, cost_bps=5)

        # Position starts flat (nothing to react to yet), enters on bar 2.
        assert res["position"].iloc[0] == 0.0
        assert (res["position"].iloc[1:] == 1.0).all()

        cost = 5 / 10_000
        expected = CAPITAL
        rets = pd.Series(closes).pct_change().dropna().to_list()
        expected *= 1 + rets[0] - cost  # entry bar pays the one entry fee
        for r in rets[1:]:
            expected *= 1 + r
        assert res["equity"].iloc[-1] == pytest.approx(expected)

    def test_zero_cost_constant_long_equals_exact_buy_and_hold(self):
        prices = make_prices([100, 92, 130, 145, 140])
        signals = pd.Series(1.0, index=prices.index)
        res = run_backtest(prices, signals, cost_bps=0)
        # missing day-1 return only: equity ratio = close[-1]/close[0]... but
        # position is flat on bar 1, whose return is 0 by construction (pct_change
        # fillna) — so the full price ratio is captured from bar 2 onward.
        assert res["equity"].iloc[-1] / CAPITAL == pytest.approx(140 / 100)

    def test_single_flip_charges_exactly_one_cost(self):
        prices = make_prices([100.0] * 10)  # flat prices isolate the fee
        signals = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], index=prices.index, dtype=float)
        res = run_backtest(prices, signals, initial_capital=CAPITAL, cost_bps=5)
        assert res["turnover"].sum() == pytest.approx(1.0)
        assert res["equity"].iloc[-1] == pytest.approx(CAPITAL * (1 - 5 / 10_000))

    def test_round_trip_charges_two_costs(self):
        prices = make_prices([100.0] * 10)
        signals = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], index=prices.index, dtype=float)
        res = run_backtest(prices, signals, cost_bps=10)
        assert res["turnover"].sum() == pytest.approx(2.0)
        assert res["equity"].iloc[-1] == pytest.approx(CAPITAL * (1 - 1e-3) * (1 - 1e-3))

    def test_long_to_short_reversal_costs_double_turnover(self):
        prices = make_prices([100.0] * 6)
        signals = pd.Series([1, 1, -1, -1, -1, -1], index=prices.index, dtype=float)
        res = run_backtest(prices, signals, cost_bps=5)
        # 0→1 entry (1 unit) plus 1→-1 reversal (2 units)
        assert res["turnover"].sum() == pytest.approx(3.0)


class TestLookAheadGuard:
    """The core proof: a signal that peeks at the close it trades on."""

    def test_peeking_signal_is_a_money_machine_only_in_the_naive_engine(self):
        # Alternating +10% / -9.09% days: buy-and-hold goes nowhere.
        closes = [100.0, 110.0] * 50
        prices = make_prices(closes)
        rets = prices["Close"].pct_change().fillna(0.0)

        # "Buy on up days": computed FROM the day's own return — information
        # you only have once the close you'd be trading at has printed.
        peeking = (rets > 0).astype(float)

        naive = run_naive_backtest_do_not_use(prices, peeking)
        honest = run_backtest(prices, peeking, cost_bps=0)

        # Naive engine: earns every +10% day, sits out every down day.
        assert naive["equity"].iloc[-1] > CAPITAL * 100
        # Honest engine: yesterday's up day predicts NOTHING here (it's
        # anti-correlated by construction) — the artifact evaporates.
        assert honest["equity"].iloc[-1] < CAPITAL
        assert honest["equity"].iloc[-1] < naive["equity"].iloc[-1] / 100

    def test_shift_means_first_bar_is_always_flat(self):
        prices = make_prices([100, 120, 110])
        signals = pd.Series(1.0, index=prices.index)
        res = run_backtest(prices, signals)
        assert res["position"].iloc[0] == 0.0
        assert res["daily_return"].iloc[0] == 0.0


class TestValidation:
    def test_misaligned_signals_raise(self):
        prices = make_prices([100, 101, 102])
        bad = pd.Series([1.0, 1.0], index=prices.index[:2])
        with pytest.raises(ValueError, match="share the price index"):
            run_backtest(prices, bad)

    def test_nan_close_raises(self):
        prices = make_prices([100, 101, 102])
        prices.loc[prices.index[1], "Close"] = np.nan
        signals = pd.Series(1.0, index=prices.index)
        with pytest.raises(ValueError, match="NaN"):
            run_backtest(prices, signals)

    def test_out_of_range_signal_raises(self):
        prices = make_prices([100, 101, 102])
        signals = pd.Series([0.0, 2.0, 0.0], index=prices.index)
        with pytest.raises(ValueError, match="signals must lie"):
            run_backtest(prices, signals)

    def test_nan_signals_treated_as_flat(self):
        prices = make_prices([100, 110, 121, 133])
        signals = pd.Series([np.nan, 1.0, np.nan, 1.0], index=prices.index)
        res = run_backtest(prices, signals, cost_bps=0)
        assert res["position"].tolist() == [0.0, 0.0, 1.0, 0.0]

    def test_negative_cost_raises(self):
        prices = make_prices([100, 101])
        signals = pd.Series(1.0, index=prices.index)
        with pytest.raises(ValueError, match="cost_bps"):
            run_backtest(prices, signals, cost_bps=-1)

    def test_series_input_accepted(self):
        closes = pd.Series([100.0, 101.0, 102.0], index=pd.bdate_range("2020-01-01", periods=3))
        signals = pd.Series(1.0, index=closes.index)
        res = run_backtest(closes, signals)
        assert len(res) == 3
