"""Trade-ledger tests on constructed position paths."""

import numpy as np
import pandas as pd
import pytest

from src.engine import run_backtest
from src.trades import extract_trades, trade_stats


def make_prices(closes):
    idx = pd.bdate_range("2021-01-01", periods=len(closes))
    s = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s, "Volume": 1e6})


class TestExtraction:
    def test_single_round_trip(self):
        closes = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        prices = make_prices(closes)
        sig = pd.Series([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], index=prices.index, dtype=float)
        res = run_backtest(prices, sig, cost_bps=0)

        trades = extract_trades(res, prices)
        assert len(trades) == 1
        t = trades.iloc[0]
        # signal fired at bar 2's close (104) and exited at bar 5's close… the
        # stance is held bars 3–5, entry decided/executed at bar 2's close,
        # exit decided at bar 5's close → priced 104 → 110.
        assert t["direction"] == 1
        assert t["entry_date"] == prices.index[2]
        assert t["entry_price"] == pytest.approx(104)
        assert t["exit_date"] == prices.index[5]
        assert t["exit_price"] == pytest.approx(110)
        assert t["bars_held"] == 3
        assert t["gross_return"] == pytest.approx(110 / 104 - 1)
        assert not t["open"]

    def test_open_trade_flagged(self):
        prices = make_prices([100, 101, 102, 103, 104])
        sig = pd.Series([0, 1, 1, 1, 1], index=prices.index, dtype=float)
        res = run_backtest(prices, sig, cost_bps=0)
        trades = extract_trades(res, prices)
        assert len(trades) == 1
        assert bool(trades.iloc[0]["open"])
        assert trades.iloc[0]["exit_date"] == prices.index[-1]

    def test_reversal_creates_two_trades(self):
        prices = make_prices([100, 100, 100, 100, 100, 100])
        sig = pd.Series([1, 1, -1, -1, 0, 0], index=prices.index, dtype=float)
        res = run_backtest(prices, sig, cost_bps=0)
        trades = extract_trades(res, prices)
        assert trades["direction"].tolist() == [1, -1]

    def test_short_profits_when_price_falls(self):
        closes = [100, 100, 90, 81, 81, 81]
        prices = make_prices(closes)
        sig = pd.Series([-1, -1, -1, 0, 0, 0], index=prices.index, dtype=float)
        res = run_backtest(prices, sig, cost_bps=0)
        trades = extract_trades(res, prices)
        assert len(trades) == 1
        t = trades.iloc[0]
        assert t["direction"] == -1
        # short over two -10% days, compounded daily: (1.1)(1.1) - 1 = 21%
        assert t["gross_return"] == pytest.approx(1.1 * 1.1 - 1)

    def test_no_trades_on_flat_signal(self):
        prices = make_prices([100, 101, 102])
        sig = pd.Series(0.0, index=prices.index)
        res = run_backtest(prices, sig)
        assert len(extract_trades(res, prices)) == 0

    def test_ledger_matches_engine_equity_when_costless(self):
        # Compounding every trade's gross return should reproduce the
        # engine's final equity (cost-free, long-only, no open position).
        rng = np.random.default_rng(11)
        closes = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, 120))
        prices = make_prices(closes)
        sig = pd.Series((np.arange(120) // 10) % 2, index=prices.index, dtype=float)
        sig.iloc[-10:] = 0.0  # end flat so no trade is left open
        res = run_backtest(prices, sig, cost_bps=0)
        trades = extract_trades(res, prices)
        assert not trades["open"].any()
        compounded = (1 + trades["gross_return"]).prod()
        assert res["equity"].iloc[-1] / res["equity"].iloc[0] == pytest.approx(compounded)


class TestStats:
    def test_hand_computed_stats(self):
        trades = pd.DataFrame(
            {
                "gross_return": [0.10, -0.05, 0.20, -0.10],
                "bars_held": [5, 3, 10, 2],
                "open": [False] * 4,
            }
        )
        s = trade_stats(trades)
        assert s["Trades"] == 4
        assert s["Win rate (per trade)"] == pytest.approx(0.5)
        assert s["Avg win"] == pytest.approx(0.15)
        assert s["Avg loss"] == pytest.approx(-0.075)
        assert s["Profit factor"] == pytest.approx(0.30 / 0.15)
        assert s["Avg bars held"] == pytest.approx(5.0)
        assert s["Best trade"] == pytest.approx(0.20)
        assert s["Worst trade"] == pytest.approx(-0.10)

    def test_empty_ledger(self):
        assert trade_stats(pd.DataFrame(columns=["gross_return"]))["Trades"] == 0

    def test_no_losses_profit_factor_infinite(self):
        trades = pd.DataFrame({"gross_return": [0.1, 0.2], "bars_held": [1, 2], "open": [False, False]})
        assert trade_stats(trades)["Profit factor"] == float("inf")


class TestMidPositionFrames:
    def test_frame_starting_in_position_prices_entry_at_first_bar(self):
        # An out-of-sample slice can begin mid-trade; the ledger must not wrap
        # to index -1 and price the entry off the final (future) bar.
        idx = pd.bdate_range("2021-01-01", periods=5)
        closes = pd.Series([100.0, 110.0, 121.0, 121.0, 121.0], index=idx)
        prices = pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                               "Close": closes, "Volume": 1e6})
        results = pd.DataFrame(
            {
                "position": [1.0, 1.0, 1.0, 0.0, 0.0],
                "asset_return": closes.pct_change().fillna(0.0),
            },
            index=idx,
        )
        trades = extract_trades(results, prices)
        assert len(trades) == 1
        t = trades.iloc[0]
        assert t["entry_date"] == idx[0]
        assert t["entry_price"] == pytest.approx(100.0)
        assert t["exit_price"] == pytest.approx(121.0)
