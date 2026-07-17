import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd


BACKTEST_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = BACKTEST_DIR.parent
PROJECT_ROOT = CHAPTER_DIR.parents[1]
CHAPTER_RESULTS_DIR = PROJECT_ROOT / "results" / "chapter2"

FACTOR_ID = "22"
FACTOR_NAME = "high_bias_oi_drop_hourly"
FACTOR_LABEL = f"{FACTOR_ID}_{FACTOR_NAME}"
FACTOR_SCRIPT = CHAPTER_DIR / "factors" / f"{FACTOR_LABEL}.py"

DEFAULT_HOURLY_DIR = CHAPTER_RESULTS_DIR / "tables" / "hourly"
DEFAULT_OUTPUT_DIR = (
    CHAPTER_RESULTS_DIR / "backtest" / f"{FACTOR_LABEL}_simple"
)
BACKTEST_START_DATE = pd.Timestamp("2021-01-01")
BACKTEST_END_DATE = pd.Timestamp("2026-12-31")
TRADING_DAYS_PER_YEAR = 252
DEFAULT_TRADING_BARS_PER_DAY = 9
TRADING_BARS_PER_YEAR = TRADING_DAYS_PER_YEAR * DEFAULT_TRADING_BARS_PER_DAY

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple hourly backtest for factor 22."
    )
    parser.add_argument("--hourly-dir", type=Path, default=DEFAULT_HOURLY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def load_factor_module():
    spec = importlib.util.spec_from_file_location("factor_22_hourly", FACTOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_symbols(factor, hourly_dir, symbols):
    all_symbols = factor.discover_symbols(hourly_dir)
    if symbols is None:
        return all_symbols
    wanted = {symbol.strip().upper() for symbol in symbols}
    selected = [symbol for symbol in all_symbols if symbol in wanted]
    if not selected:
        raise ValueError(f"No selected symbols found: {','.join(sorted(wanted))}")
    return selected


def add_curves(frame, return_column, prefix):
    frame[f"{prefix}_cumulative_return"] = frame[return_column].fillna(0).cumsum()
    frame[f"{prefix}_equity"] = 1 + frame[f"{prefix}_cumulative_return"]
    frame[f"{prefix}_drawdown"] = (
        frame[f"{prefix}_equity"] / frame[f"{prefix}_equity"].cummax() - 1
    )


def prepare_backtest_frame(factor, symbol, hourly_dir):
    hourly = factor.load_hourly(symbol, hourly_dir)
    hourly = hourly[
        hourly["trading_date"].between(BACKTEST_START_DATE, BACKTEST_END_DATE)
    ].copy()
    if hourly.empty:
        raise ValueError(f"{symbol} has no rows in backtest date window.")

    frame = factor.build_intraday_daily_bars(hourly)
    frame = factor.add_daily_frequency_features(frame)
    frame = factor.build_short_state_machine(frame)
    frame.insert(0, "symbol", symbol)

    bar_open = frame["hourly_open"].replace(0, np.nan)
    prev_close = frame["close"].shift(1).replace(0, np.nan)
    prev_position = frame["position"].shift(1).fillna(0)
    gap_return = (bar_open / prev_close - 1).fillna(0)
    intraday_return = (frame["close"] / bar_open - 1).fillna(0)
    if not frame.empty:
        gap_return.iloc[0] = 0
        intraday_return.iloc[0] = 0

    frame["gap_return"] = gap_return
    frame["intraday_return"] = intraday_return
    frame["benchmark_position"] = 1
    frame["strategy_net_position"] = 1 + 2 * frame["position"]
    frame["previous_strategy_net_position"] = 1 + 2 * prev_position
    frame["benchmark_return"] = gap_return + intraday_return
    frame["strategy_return"] = (
        frame["previous_strategy_net_position"] * gap_return
        + frame["strategy_net_position"] * intraday_return
    )
    frame["excess_return"] = frame["strategy_return"] - frame["benchmark_return"]
    add_curves(frame, "strategy_return", "strategy")
    add_curves(frame, "benchmark_return", "benchmark")
    add_curves(frame, "excess_return", "excess")
    return frame


def add_trade_returns(trades):
    trades = trades.copy()
    if trades.empty:
        trades["trade_return"] = pd.Series(dtype=float)
        return trades

    trades["trade_return"] = np.where(
        trades["exit_price"].notna() & trades["entry_price"].notna(),
        trades["entry_price"] / trades["exit_price"] - 1,
        np.nan,
    )
    return trades


def annualized_return(
    total_return,
    period_count,
    periods_per_year=TRADING_BARS_PER_YEAR,
):
    if period_count <= 0:
        return np.nan
    return total_return / period_count * periods_per_year


def infer_trading_bars_per_year(frame):
    trading_days = frame["trading_date"].nunique()
    if trading_days <= 0:
        return TRADING_BARS_PER_YEAR
    return TRADING_DAYS_PER_YEAR * len(frame) / trading_days


def summarize(frame, trades):
    closed = trades[trades["status"] == "closed"]
    bars_per_year = infer_trading_bars_per_year(frame)
    strategy_total_return = frame["strategy_cumulative_return"].iloc[-1]
    benchmark_total_return = frame["benchmark_cumulative_return"].iloc[-1]
    excess_total_return = frame["excess_cumulative_return"].iloc[-1]
    return {
        "symbol": frame["symbol"].iloc[0],
        "start_date": frame["date"].min(),
        "end_date": frame["date"].max(),
        "total_bars": len(frame),
        "trading_days": frame["trading_date"].nunique(),
        "bars_per_year": bars_per_year,
        "signal_count": int(frame["open_short_signal"].sum()),
        "entry_count": int((frame["trade_signal"] == -1).sum()),
        "exit_count": int((frame["trade_signal"] == 1).sum()),
        "holding_bars": int((frame["position"] == -1).sum()),
        "mean_strategy_net_position": frame["strategy_net_position"].mean(),
        "strategy_total_return": strategy_total_return,
        "strategy_annual_return": annualized_return(
            strategy_total_return,
            len(frame),
            bars_per_year,
        ),
        "benchmark_total_return": benchmark_total_return,
        "benchmark_annual_return": annualized_return(
            benchmark_total_return,
            len(frame),
            bars_per_year,
        ),
        "excess_total_return": excess_total_return,
        "excess_annual_return": annualized_return(
            excess_total_return,
            len(frame),
            bars_per_year,
        ),
        "strategy_max_drawdown": frame["strategy_drawdown"].min(),
        "benchmark_max_drawdown": frame["benchmark_drawdown"].min(),
        "strategy_sharpe": annualized_sharpe(
            frame["strategy_return"],
            bars_per_year,
        ),
        "benchmark_sharpe": annualized_sharpe(
            frame["benchmark_return"],
            bars_per_year,
        ),
        "excess_sharpe": annualized_sharpe(
            frame["excess_return"],
            bars_per_year,
        ),
        "trade_count": len(trades),
        "closed_trade_count": len(closed),
        "mean_trade_return": closed["trade_return"].mean(),
        "win_rate": (closed["trade_return"] > 0).mean(),
    }


def annualized_sharpe(returns, periods_per_year=TRADING_BARS_PER_YEAR):
    std = returns.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan
    return returns.mean() / std * np.sqrt(periods_per_year)


def plot_equity(frame, path, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame["date"], frame["strategy_cumulative_return"], label="strategy")
    ax.plot(frame["date"], frame["benchmark_cumulative_return"], label="benchmark")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_empty(path, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        "No symbols with signals",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_return_summary(summaries, path):
    plot_data = summaries.copy()
    required_columns = {
        "symbol",
        "strategy_total_return",
        "benchmark_total_return",
        "excess_total_return",
    }
    missing_columns = required_columns - set(plot_data.columns)
    if missing_columns:
        raise ValueError(
            "Summary table missing columns for return plot: "
            f"{','.join(sorted(missing_columns))}"
        )

    if "signal_count" in plot_data.columns:
        signal_count = pd.to_numeric(
            plot_data["signal_count"],
            errors="coerce",
        ).fillna(0)
        plot_data = plot_data[signal_count > 0].copy()

    plot_data = plot_data.sort_values("excess_total_return", ascending=True)
    labels = plot_data["symbol"].replace(
        {"ALL_SYMBOLS_EQUAL_WEIGHT": "ALL_EQUAL_WEIGHT"}
    )
    y_positions = np.arange(len(plot_data))
    bar_height = 0.25
    figure_height = max(6, min(24, len(plot_data) * 0.35 + 2))

    fig, ax = plt.subplots(figsize=(12, figure_height))
    ax.barh(
        y_positions - bar_height,
        plot_data["benchmark_total_return"],
        height=bar_height,
        color="#ff7f0e",
        alpha=0.78,
        label="buy-and-hold benchmark",
    )
    ax.barh(
        y_positions,
        plot_data["strategy_total_return"],
        height=bar_height,
        color="#1f77b4",
        alpha=0.86,
        label="strategy: long, reverse short on signal",
    )
    ax.barh(
        y_positions + bar_height,
        plot_data["excess_total_return"],
        height=bar_height,
        color="#2ca02c",
        alpha=0.74,
        label="excess",
    )
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Total return")
    ax.set_title(f"{FACTOR_LABEL} total return summary")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run_symbol(factor, symbol, hourly_dir, output_dir):
    frame = prepare_backtest_frame(factor, symbol, hourly_dir)
    trades = add_trade_returns(factor.build_trade_table(symbol, frame))
    summary = summarize(frame, trades)

    hourly_dir_out = output_dir / "tables" / "hourly"
    trade_dir = output_dir / "tables" / "trades"
    figure_dir = output_dir / "figures"
    hourly_dir_out.mkdir(parents=True, exist_ok=True)
    trade_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(
        hourly_dir_out / f"{symbol}_{FACTOR_LABEL}_simple_backtest.csv",
        index=False,
    )
    trades.to_csv(
        trade_dir / f"{symbol}_{FACTOR_LABEL}_simple_trades.csv",
        index=False,
    )
    plot_equity(
        frame,
        figure_dir / f"{symbol}_{FACTOR_LABEL}_simple_equity.png",
        f"{symbol} {FACTOR_LABEL}",
    )
    return frame, trades, summary


def build_portfolio(frames):
    merged = pd.concat(frames, ignore_index=True).sort_values(
        ["symbol", "trading_date", "date"]
    )
    symbol_daily = (
        merged.groupby(["symbol", "trading_date"], sort=False)
        .agg(
            date=("trading_date", "first"),
            last_close=("close", "last"),
            hourly_bar_count=("date", "size"),
            holding_symbol_count=(
                "position",
                lambda values: (values == -1).mean(),
            ),
            strategy_net_position=("strategy_net_position", "mean"),
            signal_count=("open_short_signal", "sum"),
            entry_count=("trade_signal", lambda values: int((values == -1).sum())),
            exit_count=("trade_signal", lambda values: int((values == 1).sum())),
            excess_return=("excess_return", "sum"),
        )
        .reset_index()
        .sort_values(["symbol", "trading_date"])
    )
    symbol_daily["benchmark_return"] = (
        symbol_daily.groupby("symbol")["last_close"]
        .pct_change(fill_method=None)
        .fillna(0)
    )
    symbol_daily["strategy_return"] = (
        symbol_daily["benchmark_return"] + symbol_daily["excess_return"]
    )

    portfolio = (
        symbol_daily.groupby("trading_date")
        .agg(
            date=("date", "first"),
            symbol_count=("symbol", "nunique"),
            mean_hourly_bar_count=("hourly_bar_count", "mean"),
            holding_symbol_count=("holding_symbol_count", "sum"),
            strategy_net_position=("strategy_net_position", "mean"),
            signal_count=("signal_count", "sum"),
            entry_count=("entry_count", "sum"),
            exit_count=("exit_count", "sum"),
            strategy_return=("strategy_return", "mean"),
            benchmark_return=("benchmark_return", "mean"),
            excess_return=("excess_return", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )
    portfolio["symbol"] = "ALL_SYMBOLS_EQUAL_WEIGHT"
    portfolio["excess_return"] = portfolio["strategy_return"] - portfolio[
        "benchmark_return"
    ]
    add_curves(portfolio, "strategy_return", "strategy")
    add_curves(portfolio, "benchmark_return", "benchmark")
    add_curves(portfolio, "excess_return", "excess")
    return portfolio


def summarize_portfolio(portfolio, summaries, trades):
    closed = trades[trades["status"] == "closed"]
    strategy_total_return = portfolio["strategy_cumulative_return"].iloc[-1]
    benchmark_total_return = portfolio["benchmark_cumulative_return"].iloc[-1]
    excess_total_return = portfolio["excess_cumulative_return"].iloc[-1]
    return {
        "symbol": "ALL_SYMBOLS_EQUAL_WEIGHT",
        "start_date": portfolio["date"].min(),
        "end_date": portfolio["date"].max(),
        "total_bars": len(portfolio),
        "trading_days": portfolio["trading_date"].nunique(),
        "bars_per_year": TRADING_DAYS_PER_YEAR,
        "symbol_count": summaries["symbol"].nunique(),
        "mean_bar_symbol_count": portfolio["symbol_count"].mean(),
        "mean_daily_symbol_count": portfolio["symbol_count"].mean(),
        "mean_hourly_bar_count": portfolio["mean_hourly_bar_count"].mean(),
        "mean_holding_symbol_count": portfolio["holding_symbol_count"].mean(),
        "signal_count": portfolio["signal_count"].sum(),
        "entry_count": portfolio["entry_count"].sum(),
        "exit_count": portfolio["exit_count"].sum(),
        "mean_strategy_net_position": portfolio["strategy_net_position"].mean(),
        "strategy_total_return": strategy_total_return,
        "strategy_annual_return": annualized_return(
            strategy_total_return,
            len(portfolio),
            TRADING_DAYS_PER_YEAR,
        ),
        "benchmark_total_return": benchmark_total_return,
        "benchmark_annual_return": annualized_return(
            benchmark_total_return,
            len(portfolio),
            TRADING_DAYS_PER_YEAR,
        ),
        "excess_total_return": excess_total_return,
        "excess_annual_return": annualized_return(
            excess_total_return,
            len(portfolio),
            TRADING_DAYS_PER_YEAR,
        ),
        "strategy_max_drawdown": portfolio["strategy_drawdown"].min(),
        "benchmark_max_drawdown": portfolio["benchmark_drawdown"].min(),
        "strategy_sharpe": annualized_sharpe(
            portfolio["strategy_return"],
            TRADING_DAYS_PER_YEAR,
        ),
        "benchmark_sharpe": annualized_sharpe(
            portfolio["benchmark_return"],
            TRADING_DAYS_PER_YEAR,
        ),
        "excess_sharpe": annualized_sharpe(
            portfolio["excess_return"],
            TRADING_DAYS_PER_YEAR,
        ),
        "trade_count": len(trades),
        "closed_trade_count": len(closed),
        "mean_trade_return": closed["trade_return"].mean(),
        "win_rate": (closed["trade_return"] > 0).mean(),
    }


def filter_signal_summaries(summaries):
    signal_count = pd.to_numeric(
        summaries["signal_count"],
        errors="coerce",
    ).fillna(0)
    return summaries[signal_count > 0].copy()


def build_signal_plot_data(frames, summaries, trades):
    plot_summaries = filter_signal_summaries(summaries)
    signal_symbols = set(plot_summaries["symbol"])
    plot_frames = [
        frame for frame in frames if frame["symbol"].iloc[0] in signal_symbols
    ]
    plot_trades = trades[trades["symbol"].isin(signal_symbols)].copy()
    plot_portfolio = build_portfolio(plot_frames) if plot_frames else None
    return plot_summaries, plot_trades, plot_portfolio


def save_all_outputs(
    output_dir,
    summaries,
    trades,
    portfolio,
    plot_summaries,
    plot_trades,
    plot_portfolio,
):
    summary_dir = output_dir / "tables" / "summary"
    trade_dir = output_dir / "tables" / "trades"
    portfolio_dir = output_dir / "tables" / "portfolio"
    figure_dir = output_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    trade_dir.mkdir(parents=True, exist_ok=True)
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = pd.concat(
        [
            summaries,
            pd.DataFrame([summarize_portfolio(portfolio, summaries, trades)]),
        ],
        ignore_index=True,
        sort=False,
    )

    all_summaries.to_csv(
        summary_dir / f"all_symbols_{FACTOR_LABEL}_simple_summary.csv",
        index=False,
    )
    trades.to_csv(
        trade_dir / f"all_symbols_{FACTOR_LABEL}_simple_trades.csv",
        index=False,
    )
    portfolio.to_csv(
        portfolio_dir / f"all_symbols_{FACTOR_LABEL}_simple_equal_weight.csv",
        index=False,
    )

    signal_equity_path = (
        figure_dir / f"all_symbols_{FACTOR_LABEL}_simple_equal_weight_equity.png"
    )
    signal_return_path = (
        figure_dir / f"all_symbols_{FACTOR_LABEL}_simple_return_summary.png"
    )
    legacy_signal_return_path = (
        figure_dir / f"all_symbols_{FACTOR_LABEL}_simple_annual_return_summary.png"
    )
    if plot_portfolio is None:
        plot_empty(signal_equity_path, f"Signal symbols equal weight {FACTOR_LABEL}")
        plot_return_summary(plot_summaries, signal_return_path)
        plot_return_summary(plot_summaries, legacy_signal_return_path)
        return

    plot_all_summaries = pd.concat(
        [
            plot_summaries,
            pd.DataFrame(
                [summarize_portfolio(plot_portfolio, plot_summaries, plot_trades)]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    plot_equity(
        plot_portfolio,
        signal_equity_path,
        f"Signal symbols equal weight {FACTOR_LABEL}",
    )
    plot_return_summary(plot_all_summaries, signal_return_path)
    plot_return_summary(plot_all_summaries, legacy_signal_return_path)


def main():
    args = parse_args()
    hourly_dir = args.hourly_dir.resolve()
    output_dir = args.output_dir.resolve()
    factor = load_factor_module()
    symbols = select_symbols(factor, hourly_dir, args.symbol)

    frames = []
    trades = []
    summaries = []
    failures = []
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hourly data directory: {hourly_dir}", flush=True)
    print(f"Backtest output directory: {output_dir}", flush=True)
    print(
        f"Backtest date window: {BACKTEST_START_DATE.date()} to "
        f"{BACKTEST_END_DATE.date()}",
        flush=True,
    )
    print(f"Symbols to backtest: {len(symbols)}", flush=True)

    for symbol in symbols:
        try:
            frame, symbol_trades, summary = run_symbol(
                factor,
                symbol,
                hourly_dir,
                output_dir,
            )
            frames.append(frame)
            trades.append(symbol_trades)
            summaries.append(summary)
            print(
                f"[{symbol}] strategy: {summary['strategy_total_return']:.6f}; "
                f"strategy annual: {summary['strategy_annual_return']:.6f}; "
                f"benchmark: {summary['benchmark_total_return']:.6f}; "
                f"benchmark annual: {summary['benchmark_annual_return']:.6f}; "
                f"trades: {summary['trade_count']}",
                flush=True,
            )
        except Exception as exc:
            if not args.keep_going:
                raise
            failures.append((symbol, exc))
            print(f"[{symbol}] failed: {exc}", flush=True)

    if not summaries:
        raise RuntimeError("No symbols were backtested successfully.")

    summary_table = pd.DataFrame(summaries).sort_values("symbol")
    non_empty_trades = [trade for trade in trades if not trade.empty]
    trade_table = (
        pd.concat(non_empty_trades, ignore_index=True, sort=False)
        if non_empty_trades
        else trades[0].copy()
    )
    portfolio = build_portfolio(frames)
    plot_summaries, plot_trades, plot_portfolio = build_signal_plot_data(
        frames,
        summary_table,
        trade_table,
    )
    save_all_outputs(
        output_dir,
        summary_table,
        trade_table,
        portfolio,
        plot_summaries,
        plot_trades,
        plot_portfolio,
    )

    print("Backtest complete.", flush=True)
    print(f"Successful symbols: {len(summaries)}", flush=True)

    if failures:
        print(f"Failed symbols: {len(failures)}", flush=True)
        for symbol, exc in failures:
            print(f"- {symbol}: {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
