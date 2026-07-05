"""Data-layer tests: cache behavior offline (fake yfinance), live fetch marked."""

import sys
import types

import pandas as pd
import pytest

from src.data import fetch_prices


def make_yf_frame(start="2020-01-01", periods=300, multiindex=True):
    idx = pd.bdate_range(start, periods=periods)
    data = {
        "Close": pd.Series(range(100, 100 + periods), index=idx, dtype=float),
        "High": 1.0, "Low": 1.0, "Open": 1.0, "Volume": 1e6,
    }
    df = pd.DataFrame(data, index=idx)
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, ["TEST"]], names=["Price", "Ticker"])
    return df


@pytest.fixture
def fake_yf(monkeypatch):
    """Replace the yfinance module with a canned, call-counting fake."""
    calls = []

    def download(ticker, start=None, end=None, **kwargs):
        calls.append((str(ticker), pd.Timestamp(start), pd.Timestamp(end)))
        if ticker == "BADTICKER":
            return pd.DataFrame()
        idx = pd.bdate_range(start, end)
        if len(idx) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(
            {"Close": 100.0, "High": 101.0, "Low": 99.0, "Open": 100.0, "Volume": 1e6},
            index=idx,
        )
        df.columns = pd.MultiIndex.from_product([df.columns, [ticker]], names=["Price", "Ticker"])
        return df

    fake = types.ModuleType("yfinance")
    fake.download = download
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    return calls


class TestFetchPrices:
    def test_downloads_flattens_and_caches(self, fake_yf, tmp_path):
        df = fetch_prices("TEST", "2020-01-01", "2020-06-01", cache_dir=tmp_path)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert not isinstance(df.columns, pd.MultiIndex)
        assert len(fake_yf) == 1
        assert (tmp_path / "TEST.parquet").exists()
        assert (tmp_path / "TEST.meta.json").exists()

    def test_second_call_hits_cache(self, fake_yf, tmp_path):
        fetch_prices("TEST", "2020-01-01", "2020-06-01", cache_dir=tmp_path)
        df = fetch_prices("TEST", "2020-01-01", "2020-06-01", cache_dir=tmp_path)
        assert len(fake_yf) == 1  # no second network call
        assert df is not None

    def test_narrower_range_hits_cache(self, fake_yf, tmp_path):
        fetch_prices("TEST", "2020-01-01", "2020-12-01", cache_dir=tmp_path)
        df = fetch_prices("TEST", "2020-03-01", "2020-06-01", cache_dir=tmp_path)
        assert len(fake_yf) == 1
        assert df.index.min() >= pd.Timestamp("2020-03-01")
        assert df.index.max() < pd.Timestamp("2020-06-01")

    def test_wider_range_refetches_union(self, fake_yf, tmp_path):
        fetch_prices("TEST", "2020-03-01", "2020-06-01", cache_dir=tmp_path)
        fetch_prices("TEST", "2020-01-01", "2020-09-01", cache_dir=tmp_path)
        assert len(fake_yf) == 2
        _, start, end = fake_yf[-1]
        assert start == pd.Timestamp("2020-01-01")
        assert end == pd.Timestamp("2020-09-01")

    def test_force_refresh_bypasses_cache(self, fake_yf, tmp_path):
        fetch_prices("TEST", "2020-01-01", "2020-06-01", cache_dir=tmp_path)
        fetch_prices("TEST", "2020-01-01", "2020-06-01", cache_dir=tmp_path, force_refresh=True)
        assert len(fake_yf) == 2

    def test_bad_ticker_returns_none(self, fake_yf, tmp_path):
        assert fetch_prices("BADTICKER", "2020-01-01", "2020-06-01", cache_dir=tmp_path) is None

    def test_backwards_range_returns_none(self, fake_yf, tmp_path):
        assert fetch_prices("TEST", "2020-06-01", "2020-01-01", cache_dir=tmp_path) is None
        assert len(fake_yf) == 0


@pytest.mark.network
class TestLive:
    def test_spy_has_a_decade_of_bars(self, tmp_path):
        df = fetch_prices("SPY", "2015-01-01", "2025-01-01", cache_dir=tmp_path)
        assert df is not None
        assert len(df) > 2400  # ~252 bars × 10 years, minus holidays
        assert df["Close"].isna().sum() == 0
