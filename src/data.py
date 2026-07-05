"""Price-data layer: yfinance download with a local parquet cache.

Prices are *dividend/split-adjusted* (yfinance ``auto_adjust=True``), so a
buy-and-hold equity curve on ``Close`` approximates total return. All frames
are daily OHLCV with a tz-naive ``DatetimeIndex`` and single-level columns
``Open, High, Low, Close, Volume``.

Caching: each ticker gets ``data/cache/{TICKER}.parquet`` plus a sidecar
``{TICKER}.meta.json`` recording the date range actually *requested* from
Yahoo. A new request is served from cache when its range is inside the
cached range; otherwise the union of both ranges is refetched and the cache
overwritten. The sidecar matters because the first bar of a cached frame can
sit days after the requested start (holidays, weekends, IPO date) — comparing
against bar dates alone would refetch forever.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_paths(ticker: str, cache_dir: Path) -> tuple[Path, Path]:
    safe = re.sub(r"[^A-Za-z0-9_.^=-]", "_", ticker.upper())
    return cache_dir / f"{safe}.parquet", cache_dir / f"{safe}.meta.json"


def _normalize(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Flatten yfinance's MultiIndex columns and standardize the frame."""
    if raw is None or len(raw) == 0:
        return None
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in COLUMNS if c in df.columns]]
    if "Close" not in df.columns:
        return None
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    return df if len(df) else None


def fetch_prices(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame | None:
    """Return daily OHLCV for ``ticker`` over ``[start, end)``, or ``None``.

    Serves from the parquet cache when the requested range was already
    fetched; otherwise downloads (expanding to the union of old + new range)
    and overwrites the cache. Invalid tickers and empty responses log a
    warning and return ``None`` instead of raising.
    """
    cache_dir = Path(cache_dir)
    pq_path, meta_path = _cache_paths(ticker, cache_dir)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts >= end_ts:
        logger.warning("fetch_prices(%s): start %s >= end %s", ticker, start, end)
        return None

    fetch_start, fetch_end = start_ts, end_ts
    if not force_refresh and pq_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            cached_start = pd.Timestamp(meta["start"])
            cached_end = pd.Timestamp(meta["end"])
            if cached_start <= start_ts and cached_end >= end_ts:
                df = pd.read_parquet(pq_path)
                logger.debug("fetch_prices(%s): cache hit", ticker)
                return _slice(df, start_ts, end_ts, ticker)
            # Partial overlap: refetch the union so the cache only grows.
            fetch_start = min(start_ts, cached_start)
            fetch_end = max(end_ts, cached_end)
        except (KeyError, ValueError, OSError) as exc:
            logger.warning("fetch_prices(%s): unreadable cache (%s), refetching", ticker, exc)

    try:
        import yfinance as yf

        raw = yf.download(
            ticker,
            start=fetch_start,
            end=fetch_end,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:  # yfinance raises many transport-level types
        logger.warning("fetch_prices(%s): download failed: %s", ticker, exc)
        return None

    df = _normalize(raw)
    if df is None:
        logger.warning("fetch_prices(%s): no data returned (bad ticker or empty range)", ticker)
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq_path)
    meta_path.write_text(
        json.dumps({"ticker": ticker.upper(), "start": str(fetch_start.date()), "end": str(fetch_end.date())})
    )
    return _slice(df, start_ts, end_ts, ticker)


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, ticker: str) -> pd.DataFrame | None:
    out = df.loc[(df.index >= start) & (df.index < end)]
    if len(out) == 0:
        logger.warning("fetch_prices(%s): no bars in requested window", ticker)
        return None
    return out.copy()
