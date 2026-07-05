"""QuantLab — an honest algorithmic-trading backtester.

Run from the repo root:  streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.data import fetch_prices
from src.engine import run_backtest, run_naive_backtest_do_not_use
from src.evaluate import (
    comparison_table,
    evaluate_strategy,
    format_metric,
    plot_drawdown,
    plot_equity_curve,
)
from src.metrics import alpha_beta, information_ratio, rolling_sharpe
from src.sizing import volatility_target
from src.stats import block_bootstrap_sharpe, expected_max_sharpe
from src.strategies import STRATEGY_REGISTRY, STRATEGY_SPECS
from src.style import BENCHMARK_COLOR, MUTED_INK, NEGATIVE_COLOR, STRATEGY_COLOR, base_layout
from src.sweep import best_in_sample, oos_rank_of_is_best, parameter_sweep, sweep_heatmap_pair
from src.trades import extract_trades, trade_stats
from src.walkforward import plot_walk_forward, walk_forward

st.set_page_config(page_title="QuantLab", page_icon="📉", layout="wide")

# Grids for the sweep / walk-forward tabs: (x-param, y-param, full grid)
SWEEP_GRIDS = {
    "MA Crossover": ("fast", "slow", {"fast": [5, 10, 15, 20, 30, 40, 50, 60],
                                      "slow": [30, 50, 75, 100, 125, 150, 200, 250]}),
    "RSI Mean-Reversion": ("period", "oversold", {"period": [5, 7, 10, 14, 21, 28],
                                                  "oversold": [15, 20, 25, 30, 35, 40]}),
    "Bollinger Breakout": ("window", "num_std", {"window": [10, 15, 20, 30, 40, 60],
                                                 "num_std": [1.0, 1.5, 2.0, 2.5, 3.0]}),
}
WF_GRIDS = {
    "MA Crossover": {"fast": [10, 20, 50], "slow": [50, 100, 200]},
    "RSI Mean-Reversion": {"period": [7, 14, 21], "oversold": [20, 30], "overbought": [70, 80]},
    "Bollinger Breakout": {"window": [10, 20, 40], "num_std": [1.5, 2.0, 2.5]},
}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    return fetch_prices(ticker, start, end)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_sweep(ticker, start, end, strategy, train_frac, cost_bps):
    prices = load_prices(ticker, start, end)
    _, _, grid = SWEEP_GRIDS[strategy]
    return parameter_sweep(strategy, prices, grid, train_frac=train_frac, cost_bps=cost_bps)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_walkforward(ticker, start, end, strategy, n_windows, cost_bps):
    prices = load_prices(ticker, start, end)
    return walk_forward(strategy, prices, WF_GRIDS[strategy], n_windows=n_windows, cost_bps=cost_bps)


def sidebar() -> dict:
    st.sidebar.title("📉 QuantLab")
    st.sidebar.caption("An honest backtester: costs charged, signals delayed, out-of-sample respected.")

    ticker = st.sidebar.text_input("Ticker", "SPY").strip().upper()
    col1, col2 = st.sidebar.columns(2)
    start = col1.date_input("Start", pd.Timestamp("2015-01-01"), min_value=pd.Timestamp("1990-01-01"))
    end = col2.date_input("End", pd.Timestamp("2025-01-01"))

    strategy = st.sidebar.selectbox("Strategy", list(STRATEGY_SPECS))
    spec = STRATEGY_SPECS[strategy]
    st.sidebar.caption(spec.description)

    params = {}
    for p in spec.params:
        if p.kind == "bool":
            params[p.name] = st.sidebar.checkbox(p.label, value=bool(p.default), help=p.help or None)
        elif p.kind == "float":
            params[p.name] = st.sidebar.slider(
                p.label, float(p.min), float(p.max), float(p.default), step=float(p.step), help=p.help or None
            )
        else:
            params[p.name] = st.sidebar.slider(
                p.label, int(p.min), int(p.max), int(p.default), step=int(p.step), help=p.help or None
            )

    st.sidebar.divider()
    train_frac = st.sidebar.slider(
        "In-sample fraction", 0.5, 0.9, 0.7, 0.05,
        help="Chronological split: the first X% of the window is in-sample; the rest is held out.",
    )
    split_date = None
    with st.sidebar.expander("Exact split date (optional)"):
        use_exact = st.checkbox("Override the fraction with a date")
        if use_exact:
            picked = st.date_input("First out-of-sample day", pd.Timestamp("2022-01-01"))
            split_date = str(picked)
    cost_bps = st.sidebar.number_input(
        "Transaction cost (bps per trade)", 0.0, 100.0, 5.0, 1.0,
        help="Basis points charged on every position change. 5 bps ≈ liquid US ETF; try 0 to see the fantasy version.",
    )
    rf = st.sidebar.number_input("Risk-free rate (annual %)", 0.0, 10.0, 0.0, 0.25) / 100.0

    vol_target = None
    use_vt = st.sidebar.checkbox(
        "Volatility targeting", value=False,
        help="Scale the position by target ÷ realized volatility (20-day), capped at 1× — "
             "risk decides the size, not conviction. Applies to the Backtest tab.",
    )
    if use_vt:
        vol_target = st.sidebar.slider("Target volatility (annual %)", 5, 25, 10) / 100.0

    cfg = dict(
        ticker=ticker, start=str(start), end=str(end), strategy=strategy,
        params=params, train_frac=train_frac, split_date=split_date,
        cost_bps=float(cost_bps), rf=float(rf), vol_target=vol_target,
    )

    st.sidebar.divider()
    run = st.sidebar.button("▶ Run backtest", type="primary", width="stretch")
    # First load runs the defaults; afterwards, changes apply on click.
    if run or "cfg" not in st.session_state:
        st.session_state.cfg = cfg
    elif cfg != st.session_state.cfg:
        st.sidebar.caption("Settings changed — press **Run backtest** to apply.")
    return st.session_state.cfg


def render_backtest(cfg: dict, prices: pd.DataFrame) -> None:
    strategy = cfg["strategy"]
    if cfg.get("vol_target"):
        base_fn, vt = STRATEGY_REGISTRY[cfg["strategy"]], cfg["vol_target"]

        def strategy(prices_, __fn=base_fn, __vt=vt, **p):
            return volatility_target(__fn(prices_, **p), prices_, target_vol=__vt)

        strategy.__name__ = f"{cfg['strategy']} · vol-targeted {vt:.0%}"

    try:
        result = evaluate_strategy(
            strategy, prices, train_frac=cfg["train_frac"],
            cost_bps=cfg["cost_bps"], rf=cfg["rf"], split_date=cfg.get("split_date"),
            **cfg["params"],
        )
    except ValueError as exc:
        st.error(f"Invalid configuration: {exc}")
        return

    (st.warning if result.overfit else st.success)(result.verdict)

    st.plotly_chart(plot_equity_curve(result), width="stretch")
    st.plotly_chart(plot_drawdown(result), width="stretch")

    st.subheader("Metrics — in-sample vs out-of-sample")
    st.caption(
        "The out-of-sample columns are the ones that matter: parameters were "
        "never tuned on that data. A big in-sample edge that vanishes here is overfitting."
    )
    st.dataframe(comparison_table(result), width="stretch")

    oos_mask = result.results.index >= result.split_date
    oos_rets = result.results.loc[oos_mask, "daily_return"]
    bench_oos = result.benchmark.loc[oos_mask, "daily_return"]

    st.subheader("Strategy vs the market (out-of-sample CAPM)")
    capm = alpha_beta(oos_rets, bench_oos, rf=cfg["rf"])
    ir = information_ratio(oos_rets, bench_oos)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beta", f"{capm['beta']:.2f}" if pd.notna(capm["beta"]) else "—",
              help="Market sensitivity: 1 = moves with the market, 0 = market-neutral.")
    c2.metric("Alpha (ann.)", f"{capm['alpha_ann']:+.1%}" if pd.notna(capm["alpha_ann"]) else "—",
              help="Return unexplained by market exposure — the number active managers are paid for.")
    c3.metric("R²", f"{capm['r2']:.2f}" if pd.notna(capm["r2"]) else "—",
              help="How much of the strategy's variance the market explains.")
    c4.metric("Information ratio", f"{ir:.2f}" if pd.notna(ir) else "—",
              help="Active return over tracking error — Sharpe measured against the benchmark instead of cash.")

    with st.expander("Rolling 1-year Sharpe (how the 'one number' drifts)"):
        roll_s = rolling_sharpe(result.results["daily_return"], rf=cfg["rf"])
        roll_b = rolling_sharpe(result.benchmark["daily_return"], rf=cfg["rf"])
        import plotly.graph_objects as go

        rfig = go.Figure()
        rfig.add_trace(go.Scatter(x=roll_b.index, y=roll_b, name="Buy & hold", mode="lines",
                                  line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dash")))
        rfig.add_trace(go.Scatter(x=roll_s.index, y=roll_s, name=result.strategy_name, mode="lines",
                                  line=dict(color=STRATEGY_COLOR, width=2)))
        rfig.add_hline(y=0, line_color=MUTED_INK, line_width=1)
        rfig.update_layout(**base_layout("Trailing 252-day Sharpe", y_title="Sharpe", height=340))
        st.plotly_chart(rfig, width="stretch")
        st.caption(
            "A full-period Sharpe averages over regimes; the rolling view shows the strategy "
            "living and dying with market conditions."
        )

    st.subheader("Is the out-of-sample Sharpe even real?")
    try:
        boot = block_bootstrap_sharpe(oos_rets, n_boot=1000, rf=cfg["rf"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Out-of-sample Sharpe", f"{boot.point:.2f}" if pd.notna(boot.point) else "—")
        c2.metric(f"{boot.level:.0%} bootstrap CI", f"[{boot.lo:.2f}, {boot.hi:.2f}]")
        c3.metric("P(Sharpe ≤ 0)", f"{boot.p_leq_zero:.0%}")
        if boot.straddles_zero():
            st.info(
                "The confidence interval includes zero: on this evidence you **cannot** "
                "conclude the strategy has an edge. Most backtests never admit this."
            )
    except ValueError:
        st.caption("Out-of-sample segment too short for a bootstrap confidence interval.")

    with st.expander("Trade ledger (round trips, gross of costs)"):
        if cfg.get("vol_target"):
            st.caption(
                "Volatility targeting re-sizes the position a little every day, so discrete "
                "round-trip accounting stops being well-defined — the ledger below treats every "
                "size change as a boundary. Turn targeting off for clean per-trade stats."
            )
        trades = extract_trades(result.results, prices)
        if len(trades) == 0:
            st.write("No trades in this window.")
        else:
            stats = trade_stats(trades)
            cols = st.columns(4)
            cols[0].metric("Trades", stats["Trades"])
            cols[1].metric("Win rate (per trade)", f"{stats['Win rate (per trade)']:.0%}")
            pf = stats["Profit factor"]
            cols[2].metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
            cols[3].metric("Avg bars held", f"{stats['Avg bars held']:.0f}")
            show = trades.copy()
            show["gross_return"] = show["gross_return"].map("{:+.2%}".format)
            st.dataframe(show, width="stretch", hide_index=True)


def render_walkforward(cfg: dict, prices: pd.DataFrame) -> None:
    st.caption(
        "A single train/test split can still be gamed — tweak parameters until the one held-out "
        "stretch looks good and it has quietly become in-sample. Walk-forward re-optimizes on an "
        "expanding window and only ever counts the *next*, unseen segment. Every point on this "
        "curve is out-of-sample."
    )
    n_windows = st.slider("Walk-forward windows", 3, 8, 5)
    try:
        wf = cached_walkforward(cfg["ticker"], cfg["start"], cfg["end"], cfg["strategy"], n_windows, cfg["cost_bps"])
    except (ValueError, KeyError) as exc:
        st.error(f"Walk-forward failed: {exc}")
        return

    bench_eq = 100_000 * (1 + prices["Close"].pct_change().fillna(0.0).loc[wf.oos_returns.index]).cumprod()
    st.plotly_chart(plot_walk_forward(wf, benchmark_equity=bench_eq), width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Walk-forward OOS Sharpe", format_metric("Sharpe", wf.oos_metrics["Sharpe"]),
              help="Every bar traded with parameters chosen only on earlier data.")
    c2.metric("Hindsight Sharpe (overfit)", format_metric("Sharpe", wf.hindsight_sharpe),
              help=f"Best single combo picked on the FULL sample: {wf.hindsight_params}. "
                   "This is the number an overfitter would report.")
    c3.metric("Buy & hold Sharpe (same window)", format_metric("Sharpe", wf.benchmark_metrics["Sharpe"]))

    gap = wf.hindsight_sharpe - wf.oos_metrics["Sharpe"]
    if pd.notna(gap) and gap > 0.3:
        st.warning(
            f"Reality gap: the hindsight-optimized Sharpe ({wf.hindsight_sharpe:.2f}) beats the "
            f"walk-forward Sharpe ({wf.oos_metrics['Sharpe']:.2f}) by {gap:.2f}. That gap is what "
            "picking parameters with future knowledge buys you — and live trading doesn't sell it."
        )

    rows = [
        {
            "Test window": f"{w.test_start.date()} → {w.test_end.date()}",
            "Trained on": f"{w.train_start.date()} → {w.train_end.date()}",
            "Chosen params": ", ".join(f"{k}={v}" for k, v in w.best_params.items()),
            "Train Sharpe": f"{w.train_sharpe:.2f}",
            "Test Sharpe": f"{w.test_sharpe:.2f}" if pd.notna(w.test_sharpe) else "—",
        }
        for w in wf.windows
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "Watch the *Chosen params* column: if the 'best' parameters jump around from window to "
        "window, the optimum was noise, not signal."
    )


def render_sweep(cfg: dict, prices: pd.DataFrame) -> None:
    st.caption(
        "Every cell is a full backtest of one parameter combination. Left: performance on the "
        "in-sample segment, where an optimizer would shop. Right: the same combos on held-out "
        "data. The ★ marks the in-sample champion — see how it fares next door."
    )
    x, y, _grid = SWEEP_GRIDS[cfg["strategy"]]
    with st.spinner("Sweeping the parameter grid…"):
        df = cached_sweep(cfg["ticker"], cfg["start"], cfg["end"], cfg["strategy"],
                          cfg["train_frac"], cfg["cost_bps"])
    st.plotly_chart(sweep_heatmap_pair(df, x=x, y=y), width="stretch")

    best = best_in_sample(df)
    rank, pct = oos_rank_of_is_best(df)
    st.markdown(
        f"**The in-sample champion** ({x}={best[x]}, {y}={best[y]}, in-sample Sharpe "
        f"{best['is_sharpe']:.2f}) ranks **#{rank} of {len(df)}** combos out-of-sample "
        f"(beats {pct:.0%} of them, out-of-sample Sharpe {best['oos_sharpe']:.2f})."
    )
    n_is_obs = int((prices.index < prices.index[int(len(prices) * cfg["train_frac"])]).sum())
    luck = expected_max_sharpe(len(df), n_is_obs)
    if pd.notna(luck):
        st.markdown(
            f"**The luck yardstick:** even if all {len(df)} combos were pure noise, the *best* "
            f"in-sample Sharpe was expected to be **≈{luck:.2f}** just from sampling error "
            f"(expected-maximum-Sharpe under zero skill). A champion near that line proved "
            f"nothing — it merely won a lottery it was guaranteed to hold every ticket for."
        )
    if pct < 0.7:
        st.warning(
            "A champion that lands mid-pack on unseen data means the 'optimal' parameters were "
            "fit to noise. Reporting that in-sample number as expected performance is how "
            "backtests lie."
        )


def render_methodology(cfg: dict, prices: pd.DataFrame) -> None:
    st.markdown(
        """
### Why most backtests lie — and what this one does about it

**1. Look-ahead bias.** A signal computed from today's close cannot earn today's return —
you'd be trading at a price that printed before you had the information. The engine shifts
every signal one bar: the stance decided at close *t* earns returns from bar *t+1*.
Below, the same "buy on up days" rule run through both engines on your current data —
the naive one turns peeking into a money machine; the honest one shows the rule for what it is.

**2. Free trading.** Every position change is charged a configurable cost (default 5 bps of
turnover). Fast-trading strategies that look brilliant at zero cost routinely die at 5 bps.

**3. Overfitting.** Parameters tuned on all of history "predict" that same history. QuantLab
splits chronologically (never shuffled — the future must stay in the future), reports both
segments side by side, and flags the overfitting signature. The Walk-Forward tab goes further:
parameters are re-chosen on an expanding window and *only* unseen segments count.

**4. One lucky path.** History happened once. The bootstrap resamples blocks of daily returns
to ask how much of the measured Sharpe is edge and how much is sampling luck.
"""
    )
    rets = prices["Close"].pct_change().fillna(0.0)
    peeking = (rets > 0).astype(float)
    naive = run_naive_backtest_do_not_use(prices, peeking)
    honest = run_backtest(prices, peeking, cost_bps=cfg["cost_bps"])

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=naive.index, y=naive["equity"], name="Naive engine (peeks + trades free)",
                             line=dict(color=NEGATIVE_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=honest.index, y=honest["equity"], name="Honest engine (shifted + costs)",
                             line=dict(color=STRATEGY_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=prices.index, y=100_000 * prices["Close"] / prices["Close"].iloc[0],
                             name="Buy & hold", line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dash")))
    fig.update_layout(**base_layout(
        f'"Buy on up days" on {cfg["ticker"]} — the same rule, two engines', y_title="Equity ($, log scale)"))
    fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The red curve is what a one-line indexing mistake looks like: applying day *t*'s signal to "
        "day *t*'s own return. It is also, give or take, what many 'incredible backtest' screenshots "
        "on the internet are actually showing."
    )


def main() -> None:
    cfg = sidebar()

    prices = load_prices(cfg["ticker"], cfg["start"], cfg["end"])
    if prices is None or len(prices) < 60:
        st.error(
            f"No usable data for **{cfg['ticker']}** in {cfg['start']} → {cfg['end']}. "
            "Check the ticker symbol (Yahoo Finance format, e.g. SPY, AAPL, BTC-USD) and widen the dates."
        )
        st.stop()

    st.title("QuantLab — Algorithmic Trading Backtester")
    st.caption(
        f"**{cfg['ticker']}** · {prices.index[0].date()} → {prices.index[-1].date()} · "
        f"{len(prices):,} trading days · dividend-adjusted Yahoo Finance data · "
        f"costs {cfg['cost_bps']:.0f} bps per trade"
    )

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Backtest", "🔁 Walk-Forward", "🔥 Parameter Sweep", "📚 Methodology"])
    with tab1:
        render_backtest(cfg, prices)
    with tab2:
        render_walkforward(cfg, prices)
    with tab3:
        render_sweep(cfg, prices)
    with tab4:
        render_methodology(cfg, prices)

    st.divider()
    st.caption(
        "Built by Ares Cajas · educational project — nothing here is investment advice, "
        "and no backtest is a promise about the future."
    )


main()
