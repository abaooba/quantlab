"""Strategy tests: sane signals on constructed uptrend/downtrend/chop."""

import numpy as np
import pandas as pd
import pytest

from src.strategies import STRATEGY_REGISTRY, STRATEGY_SPECS
from src.strategies.bollinger import bollinger_bands, signal_bollinger_breakout
from src.strategies.ma_crossover import signal_ma_crossover
from src.strategies.rsi import signal_rsi_reversion, wilder_rsi


def frame(closes) -> pd.DataFrame:
    idx = pd.bdate_range("2018-01-01", periods=len(closes))
    s = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s, "Volume": 1e6})


def uptrend(n=300):
    return frame(np.linspace(100, 300, n))


def downtrend(n=300):
    return frame(np.linspace(300, 100, n))


def choppy(n=300, seed=7):
    rng = np.random.default_rng(seed)
    return frame(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))


ALL_STRATEGIES = list(STRATEGY_REGISTRY.items())


class TestRegistry:
    def test_three_builtins_registered(self):
        assert set(STRATEGY_REGISTRY) == {"MA Crossover", "RSI Mean-Reversion", "Bollinger Breakout"}

    def test_specs_carry_ui_metadata(self):
        for name, spec in STRATEGY_SPECS.items():
            assert spec.fn is STRATEGY_REGISTRY[name]
            assert spec.description
            for p in spec.params:
                assert p.min <= p.default <= p.max


class TestContract:
    """Every registered strategy honors the signal contract on any input."""

    @pytest.mark.parametrize("name,fn", ALL_STRATEGIES)
    @pytest.mark.parametrize("data_fn", [uptrend, downtrend, choppy])
    def test_signal_contract(self, name, fn, data_fn):
        prices = data_fn()
        sig = fn(prices)
        assert isinstance(sig, pd.Series)
        assert sig.index.equals(prices.index)
        assert not sig.isna().any()
        assert set(np.unique(sig)) <= {-1.0, 0.0, 1.0}


class TestMACrossover:
    def test_long_in_clean_uptrend(self):
        sig = signal_ma_crossover(uptrend(), fast=10, slow=30)
        assert (sig.iloc[60:] == 1.0).all()  # past warm-up, unambiguous trend

    def test_flat_in_clean_downtrend(self):
        sig = signal_ma_crossover(downtrend(), fast=10, slow=30)
        assert (sig.iloc[60:] == 0.0).all()

    def test_short_in_downtrend_when_allowed(self):
        sig = signal_ma_crossover(downtrend(), fast=10, slow=30, allow_short=True)
        assert (sig.iloc[60:] == -1.0).all()

    def test_warmup_is_flat(self):
        sig = signal_ma_crossover(uptrend(), fast=10, slow=30)
        assert (sig.iloc[:29] == 0.0).all()  # slow SMA undefined until bar 30

    def test_fast_must_be_less_than_slow(self):
        with pytest.raises(ValueError, match="fast window"):
            signal_ma_crossover(uptrend(), fast=50, slow=20)


class TestRSI:
    def test_wilder_rsi_bounds_and_extremes(self):
        rsi_up = wilder_rsi(uptrend()["Close"], 14).dropna()
        assert ((rsi_up >= 0) & (rsi_up <= 100)).all()
        assert (rsi_up == 100.0).all()  # no down days at all

        flat = frame([100.0] * 60)
        rsi_flat = wilder_rsi(flat["Close"], 14).dropna()
        assert (rsi_flat == 50.0).all()  # no movement → neutral by convention

    def test_enters_on_recovery_from_oversold(self):
        # Crash hard (RSI pins low), then rebound — entry on the turn.
        closes = list(np.linspace(100, 55, 40)) + list(np.linspace(55, 90, 40))
        sig = signal_rsi_reversion(frame(closes), period=14, oversold=30, overbought=70)
        assert sig.iloc[:40].sum() == 0  # never long during the fall
        assert (sig.iloc[40:] == 1.0).any()  # long at some point in the rebound

    def test_never_signals_during_warmup(self):
        sig = signal_rsi_reversion(choppy(), period=14)
        assert (sig.iloc[:14] == 0.0).all()

    def test_bad_levels_raise(self):
        with pytest.raises(ValueError):
            signal_rsi_reversion(choppy(), oversold=80, overbought=70)


class TestBollinger:
    def test_bands_ordering(self):
        mid, upper, lower = bollinger_bands(choppy()["Close"], 20, 2.0)
        valid = mid.dropna().index
        assert (upper.loc[valid] >= mid.loc[valid]).all()
        assert (mid.loc[valid] >= lower.loc[valid]).all()

    def test_breakout_entry_and_mean_exit(self):
        # Calm range → violent breakout → decay back through the mean.
        closes = [100 + 0.2 * ((i % 5) - 2) for i in range(60)]  # tight chop
        closes += list(np.linspace(101, 130, 15))  # breakout leg
        closes += list(np.linspace(130, 95, 25))  # collapse through the mid-band
        sig = signal_bollinger_breakout(frame(closes), window=20, num_std=2)
        assert (sig.iloc[:60] == 0.0).all()  # nothing to do in the range
        assert (sig.iloc[60:75] == 1.0).any()  # long during the thrust
        assert sig.iloc[-1] == 0.0  # collapse forced the exit

    def test_flat_series_never_trades(self):
        sig = signal_bollinger_breakout(frame([100.0] * 100), window=20, num_std=2)
        assert (sig == 0.0).all()

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            signal_bollinger_breakout(choppy(), window=1)
        with pytest.raises(ValueError):
            signal_bollinger_breakout(choppy(), num_std=0)
