import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


BACKTEST_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = BACKTEST_DIR.parent
PROJECT_ROOT = CHAPTER_DIR.parents[1]
CHAPTER_RESULTS_DIR = PROJECT_ROOT / "results" / "chapter2"
FACTOR_ID = "23"
FACTOR_NAME = "high_bias_oi_speculation_drop"
FACTOR_LABEL = f"{FACTOR_ID}_{FACTOR_NAME}"
DEFAULT_FACTOR_OUTPUT_DIR = (
    CHAPTER_RESULTS_DIR / "23_high_bias_oi_speculation_drop_all_symbols"
)
DEFAULT_OUTPUT_DIR = (
    CHAPTER_RESULTS_DIR
    / "backtest"
    / "23_high_bias_oi_speculation_drop_simple_interest"
)
TRADING_DAYS_PER_YEAR = 252
BACKTEST_START_DATE = pd.Timestamp("2021-01-01")
BACKTEST_END_DATE = pd.Timestamp("2026-12-31")

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
        description=(
            "Backtest factor 23: buy-and-hold benchmark, with reverse-short "
            "strategy overlay on factor signals."
        )
    )
    parser.add_argument(
        "--factor-output-dir",
        type=Path,
        default=DEFAULT_FACTOR_OUTPUT_DIR,
        help=f"Factor output directory. Default: {DEFAULT_FACTOR_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Backtest output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        help=(
            "Symbol to backtest. Can be repeated. "
            "By default all factor files are used."
        ),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a symbol fails and report failures at the end.",
    )
    return parser.parse_args()


def parse_factor_file(path):
    pattern = rf"^(.+)_{FACTOR_ID}_{FACTOR_NAME}$"
    match = re.match(pattern, path.stem)
    if match is None:
        return None

    return {
        "symbol": match.group(1).upper(),
        "path": path,
    }


def discover_factor_files(factor_output_dir, symbols=None):
    factor_dir = factor_output_dir / "tables" / "factors"
    if not factor_dir.is_dir():
        raise FileNotFoundError(
            f"Factor table directory does not exist: {factor_dir}. "
            "Run code/chapter2/run/run_23_high_bias_oi_speculation_drop_all.py first."
        )

    symbol_filter = (
        {symbol.strip().upper() for symbol in symbols}
        if symbols is not None
        else None
    )
    factor_files = []

    for path in sorted(factor_dir.glob(f"*_{FACTOR_LABEL}.csv")):
        file_info = parse_factor_file(path)
        if file_info is None:
            continue
        if symbol_filter is not None and file_info["symbol"] not in symbol_filter:
            continue
        factor_files.append(file_info)

    if not factor_files:
        if symbol_filter is None:
            raise FileNotFoundError(f"No {FACTOR_LABEL} factor files in {factor_dir}")
        raise FileNotFoundError(
            f"No {FACTOR_LABEL} factor files for symbols: "
            f"{','.join(sorted(symbol_filter))}"
        )

    return factor_files


def load_factor_daily(file_info):
    daily = pd.read_csv(file_info["path"])
    required_columns = {"date", "open", "close", "position"}
    missing_columns = required_columns - set(daily.columns)
    if missing_columns:
        raise ValueError(
            f"{file_info['path']} missing columns: "
            f"{','.join(sorted(missing_columns))}"
        )

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily = filter_backtest_window(daily, file_info)
    daily["symbol"] = file_info["symbol"]
    daily["open"] = pd.to_numeric(daily["open"], errors="coerce")
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily["position"] = (
        pd.to_numeric(daily["position"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if "daily_return" in daily.columns:
        daily["daily_return"] = pd.to_numeric(
            daily["daily_return"],
            errors="coerce",
        )
    else:
        daily["daily_return"] = daily["close"].pct_change()
    daily.loc[daily.index[0], "daily_return"] = np.nan

    if "overnight_return" in daily.columns:
        daily["overnight_return"] = pd.to_numeric(
            daily["overnight_return"],
            errors="coerce",
        )
    else:
        daily["overnight_return"] = (
            daily["open"] / daily["close"].shift(1).replace(0, np.nan) - 1
        )
    daily.loc[daily.index[0], "overnight_return"] = np.nan

    if "intraday_return" in daily.columns:
        daily["intraday_return"] = pd.to_numeric(
            daily["intraday_return"],
            errors="coerce",
        )
    else:
        daily["intraday_return"] = (
            daily["close"] / daily["open"].replace(0, np.nan) - 1
        )

    daily["benchmark_position"] = 1
    daily["benchmark_daily_return"] = daily["daily_return"].fillna(0)
    add_simple_curve_columns(daily, "benchmark_daily_return", "benchmark")

    daily["previous_close_position"] = (
        daily["position"].shift(1).fillna(0).astype(int)
    )
    daily["effective_position"] = daily["position"]
    daily["reverse_short_position"] = daily["effective_position"]
    daily["strategy_net_position"] = 1 + 2 * daily["reverse_short_position"]
    daily["short_overlay_daily_return"] = (
        2
        * (
            daily["previous_close_position"]
            * daily["overnight_return"].fillna(0)
            + daily["reverse_short_position"]
            * daily["intraday_return"].fillna(0)
        )
    )
    add_simple_curve_columns(
        daily,
        "short_overlay_daily_return",
        "short_overlay",
    )
    daily["simple_daily_return"] = (
        daily["benchmark_daily_return"] + daily["short_overlay_daily_return"]
    )
    add_simple_curve_columns(daily, "simple_daily_return", "simple")
    add_strategy_alias_columns(daily)
    daily["excess_daily_return"] = (
        daily["simple_daily_return"] - daily["benchmark_daily_return"]
    )
    daily["excess_cumulative_return"] = (
        daily["simple_cumulative_return"]
        - daily["benchmark_cumulative_return"]
    )

    return daily


def filter_backtest_window(daily, file_info):
    window_mask = daily["date"].between(BACKTEST_START_DATE, BACKTEST_END_DATE)
    daily = daily.loc[window_mask].copy().reset_index(drop=True)
    if daily.empty:
        raise ValueError(
            f"{file_info['symbol']} has no rows between "
            f"{BACKTEST_START_DATE.date()} and {BACKTEST_END_DATE.date()}"
        )
    return daily


def add_simple_curve_columns(frame, daily_return_column, prefix):
    frame[f"{prefix}_cumulative_return"] = (
        frame[daily_return_column].fillna(0).cumsum()
    )
    frame[f"{prefix}_equity"] = 1 + frame[f"{prefix}_cumulative_return"]
    frame[f"{prefix}_running_high"] = frame[f"{prefix}_equity"].cummax()
    frame[f"{prefix}_drawdown"] = (
        frame[f"{prefix}_equity"] - frame[f"{prefix}_running_high"]
    )
    frame[f"{prefix}_drawdown_ratio"] = (
        frame[f"{prefix}_equity"] / frame[f"{prefix}_running_high"] - 1
    )


def add_strategy_alias_columns(frame):
    for suffix in [
        "daily_return",
        "cumulative_return",
        "equity",
        "running_high",
        "drawdown",
        "drawdown_ratio",
    ]:
        frame[f"strategy_{suffix}"] = frame[f"simple_{suffix}"]


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def annualized_return(total_return, day_count):
    if day_count <= 0:
        return np.nan
    return total_return / day_count * TRADING_DAYS_PER_YEAR


def annualized_sharpe(daily_returns):
    std = daily_returns.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan
    return daily_returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)


def summarize_daily(daily, symbol):
    simple_daily_return = daily["simple_daily_return"].fillna(0)
    benchmark_daily_return = daily["benchmark_daily_return"].fillna(0)
    short_overlay_daily_return = daily["short_overlay_daily_return"].fillna(0)
    reverse_short_mask = daily["reverse_short_position"] != 0
    reverse_short_returns = simple_daily_return[reverse_short_mask]
    final_short_position = int(daily["position"].iloc[-1])
    trade_signal = (
        pd.to_numeric(daily.get("trade_signal", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    signal = (
        pd.to_numeric(daily.get("signal", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    return {
        "symbol": symbol,
        "factor_id": FACTOR_ID,
        "factor_name": FACTOR_NAME,
        "start_date": daily["date"].min(),
        "end_date": daily["date"].max(),
        "total_days": len(daily),
        "benchmark_holding_days": len(daily),
        "benchmark_holding_ratio": 1.0,
        "reverse_short_days": int(reverse_short_mask.sum()),
        "reverse_short_ratio": safe_ratio(int(reverse_short_mask.sum()), len(daily)),
        "holding_days": int(reverse_short_mask.sum()),
        "holding_ratio": safe_ratio(int(reverse_short_mask.sum()), len(daily)),
        "signal_days": int(signal.sum()),
        "entry_count": int((trade_signal == -1).sum()),
        "exit_count": int((trade_signal == 1).sum()),
        "final_position": final_short_position,
        "final_short_position": final_short_position,
        "final_strategy_net_position": 1 + 2 * final_short_position,
        "mean_strategy_net_position": daily["strategy_net_position"].mean(),
        "simple_total_return": daily["simple_cumulative_return"].iloc[-1],
        "simple_annual_return": annualized_return(
            daily["simple_cumulative_return"].iloc[-1],
            len(daily),
        ),
        "simple_max_drawdown": daily["simple_drawdown"].min(),
        "simple_max_drawdown_ratio": daily["simple_drawdown_ratio"].min(),
        "mean_simple_daily_return": simple_daily_return.mean(),
        "std_simple_daily_return": simple_daily_return.std(ddof=1),
        "simple_sharpe": annualized_sharpe(simple_daily_return),
        "strategy_total_return": daily["strategy_cumulative_return"].iloc[-1],
        "strategy_annual_return": annualized_return(
            daily["strategy_cumulative_return"].iloc[-1],
            len(daily),
        ),
        "strategy_max_drawdown": daily["strategy_drawdown"].min(),
        "strategy_max_drawdown_ratio": daily["strategy_drawdown_ratio"].min(),
        "mean_strategy_daily_return": daily["strategy_daily_return"].mean(),
        "std_strategy_daily_return": daily["strategy_daily_return"].std(ddof=1),
        "strategy_sharpe": annualized_sharpe(daily["strategy_daily_return"]),
        "benchmark_total_return": daily["benchmark_cumulative_return"].iloc[-1],
        "benchmark_annual_return": annualized_return(
            daily["benchmark_cumulative_return"].iloc[-1],
            len(daily),
        ),
        "benchmark_max_drawdown": daily["benchmark_drawdown"].min(),
        "benchmark_max_drawdown_ratio": daily[
            "benchmark_drawdown_ratio"
        ].min(),
        "mean_benchmark_daily_return": benchmark_daily_return.mean(),
        "std_benchmark_daily_return": benchmark_daily_return.std(ddof=1),
        "benchmark_sharpe": annualized_sharpe(benchmark_daily_return),
        "benchmark_daily_win_rate": safe_ratio(
            int((benchmark_daily_return > 0).sum()),
            len(benchmark_daily_return),
        ),
        "best_benchmark_daily_return": benchmark_daily_return.max(),
        "worst_benchmark_daily_return": benchmark_daily_return.min(),
        "short_overlay_total_return": daily[
            "short_overlay_cumulative_return"
        ].iloc[-1],
        "short_overlay_annual_return": annualized_return(
            daily["short_overlay_cumulative_return"].iloc[-1],
            len(daily),
        ),
        "mean_short_overlay_daily_return": short_overlay_daily_return.mean(),
        "std_short_overlay_daily_return": short_overlay_daily_return.std(ddof=1),
        "short_overlay_sharpe": annualized_sharpe(short_overlay_daily_return),
        "excess_total_return": daily["excess_cumulative_return"].iloc[-1],
        "excess_annual_return": annualized_return(
            daily["excess_cumulative_return"].iloc[-1],
            len(daily),
        ),
        "mean_excess_daily_return": daily["excess_daily_return"].mean(),
        "daily_win_rate_all_days": safe_ratio(
            int((simple_daily_return > 0).sum()),
            len(simple_daily_return),
        ),
        "daily_win_rate_holding_days": safe_ratio(
            int((reverse_short_returns > 0).sum()),
            len(reverse_short_returns),
        ),
        "best_simple_daily_return": simple_daily_return.max(),
        "worst_simple_daily_return": simple_daily_return.min(),
    }


def build_trade_table(daily, symbol):
    trade_signal = (
        pd.to_numeric(daily.get("trade_signal", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    daily = daily.copy()
    daily["trade_signal"] = trade_signal
    trades = []
    open_index = None
    open_row = None

    for index, row in daily.iterrows():
        if int(row["trade_signal"]) == -1 and open_index is None:
            open_index = index
            open_row = row
            continue

        if int(row["trade_signal"]) == 1 and open_index is not None:
            trades.append(
                build_trade_row(
                    daily=daily,
                    symbol=symbol,
                    open_index=open_index,
                    open_row=open_row,
                    exit_index=index,
                    exit_row=row,
                    status="closed",
                )
            )
            open_index = None
            open_row = None

    if open_index is not None:
        exit_index = len(daily) - 1
        exit_row = daily.iloc[exit_index]
        trades.append(
            build_trade_row(
                daily=daily,
                symbol=symbol,
                open_index=open_index,
                open_row=open_row,
                exit_index=exit_index,
                exit_row=exit_row,
                status="open",
            )
        )

    columns = [
        "symbol",
        "trade_id",
        "status",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "holding_days",
        "trade_simple_return",
        "trade_strategy_return",
        "trade_benchmark_return",
        "trade_short_overlay_return",
        "trade_excess_return",
        "trade_price_return",
        "exit_reason",
    ]
    if not trades:
        return pd.DataFrame(columns=columns)

    trade_table = pd.DataFrame(trades)
    trade_table["trade_id"] = range(1, len(trade_table) + 1)
    return trade_table[columns]


def build_trade_row(daily, symbol, open_index, open_row, exit_index, exit_row, status):
    return_slice = daily.iloc[open_index:exit_index + 1]
    trade_strategy_return = return_slice["simple_daily_return"].sum()
    trade_benchmark_return = return_slice["benchmark_daily_return"].sum()
    trade_short_overlay_return = return_slice["short_overlay_daily_return"].sum()
    trade_excess_return = return_slice["excess_daily_return"].sum()
    entry_price = open_row.get("entry_price", np.nan)
    if pd.isna(entry_price):
        entry_price = open_row["open"]
    if status == "open":
        exit_price = exit_row["close"]
    else:
        exit_price = exit_row.get("exit_price", np.nan)
        if pd.isna(exit_price):
            exit_price = exit_row["open"]
    trade_price_return = np.nan
    if pd.notna(entry_price) and entry_price != 0 and pd.notna(exit_price):
        trade_price_return = entry_price / exit_price - 1

    exit_reason = exit_row.get("exit_reason", "")
    if status == "open":
        exit_reason = "open_position"

    return {
        "symbol": symbol,
        "trade_id": np.nan,
        "status": status,
        "entry_date": open_row["date"],
        "exit_date": exit_row["date"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "holding_days": max(exit_index - open_index, 0),
        "trade_simple_return": trade_strategy_return,
        "trade_strategy_return": trade_strategy_return,
        "trade_benchmark_return": trade_benchmark_return,
        "trade_short_overlay_return": trade_short_overlay_return,
        "trade_excess_return": trade_excess_return,
        "trade_price_return": trade_price_return,
        "exit_reason": exit_reason,
    }


def add_trade_summary(summary, trades):
    closed_trades = trades[trades["status"] == "closed"].copy()
    summary["trade_count"] = len(trades)
    summary["closed_trade_count"] = len(closed_trades)
    summary["open_trade_count"] = int((trades["status"] == "open").sum())

    if closed_trades.empty:
        summary["trade_win_rate"] = np.nan
        summary["mean_trade_simple_return"] = np.nan
        summary["median_trade_simple_return"] = np.nan
        summary["best_trade_simple_return"] = np.nan
        summary["worst_trade_simple_return"] = np.nan
        summary["mean_trade_benchmark_return"] = np.nan
        summary["mean_trade_short_overlay_return"] = np.nan
        summary["mean_trade_excess_return"] = np.nan
        summary["mean_holding_days"] = np.nan
        return summary

    trade_returns = closed_trades["trade_simple_return"]
    summary["trade_win_rate"] = safe_ratio(
        int((trade_returns > 0).sum()),
        len(trade_returns),
    )
    summary["mean_trade_simple_return"] = trade_returns.mean()
    summary["median_trade_simple_return"] = trade_returns.median()
    summary["best_trade_simple_return"] = trade_returns.max()
    summary["worst_trade_simple_return"] = trade_returns.min()
    summary["mean_trade_benchmark_return"] = closed_trades[
        "trade_benchmark_return"
    ].mean()
    summary["mean_trade_short_overlay_return"] = closed_trades[
        "trade_short_overlay_return"
    ].mean()
    summary["mean_trade_excess_return"] = closed_trades[
        "trade_excess_return"
    ].mean()
    summary["mean_holding_days"] = closed_trades["holding_days"].mean()
    return summary


def save_symbol_outputs(daily, trades, summary, output_dir, symbol):
    daily_dir = output_dir / "tables" / "daily"
    trade_dir = output_dir / "tables" / "trades"
    summary_dir = output_dir / "tables" / "summary"
    figure_dir = output_dir / "figures"
    daily_dir.mkdir(parents=True, exist_ok=True)
    trade_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    daily_path = daily_dir / f"{symbol}_{FACTOR_LABEL}_simple_backtest.csv"
    trade_path = trade_dir / f"{symbol}_{FACTOR_LABEL}_simple_trades.csv"
    summary_path = summary_dir / f"{symbol}_{FACTOR_LABEL}_simple_summary.csv"
    figure_path = figure_dir / f"{symbol}_{FACTOR_LABEL}_simple_equity.png"

    daily.to_csv(daily_path, index=False)
    trades.to_csv(trade_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    plot_simple_equity(daily, figure_path, f"{symbol} {FACTOR_LABEL}")

    return {
        "daily_path": daily_path,
        "trade_path": trade_path,
        "summary_path": summary_path,
        "figure_path": figure_path,
    }


def plot_simple_equity(daily, figure_path, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        daily["date"],
        daily["simple_cumulative_return"],
        color="#1f77b4",
        linewidth=1.8,
        label="strategy: buy-and-hold, reverse short on signal",
    )
    if "benchmark_cumulative_return" in daily.columns:
        ax.plot(
            daily["date"],
            daily["benchmark_cumulative_return"],
            color="#ff7f0e",
            linewidth=1.5,
            alpha=0.9,
            label="buy-and-hold benchmark",
        )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def plot_empty(figure_path, title):
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
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def plot_return_summary(summaries, figure_path):
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

    if "signal_days" in plot_data.columns:
        signal_days = pd.to_numeric(
            plot_data["signal_days"],
            errors="coerce",
        ).fillna(0)
        is_portfolio_row = plot_data["symbol"] == "ALL_SYMBOLS_EQUAL_WEIGHT"
        plot_data = plot_data[(signal_days > 0) | is_portfolio_row].copy()

    if plot_data.empty:
        plot_empty(figure_path, f"{FACTOR_LABEL} return summary")
        return

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
        label="strategy: buy-and-hold, reverse short on signal",
    )
    ax.barh(
        y_positions + bar_height,
        plot_data["excess_total_return"],
        height=bar_height,
        color="#2ca02c",
        alpha=0.74,
        label="excess / reverse-short overlay",
    )
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Total return")
    ax.set_title(f"{FACTOR_LABEL} return summary")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def build_portfolio_daily(all_daily):
    merged = pd.concat(all_daily, ignore_index=True)
    portfolio = (
        merged.groupby("date")
        .agg(
            symbol_count=("symbol", "nunique"),
            holding_symbol_count=(
                "effective_position",
                lambda values: int((values != 0).sum()),
            ),
            simple_daily_return=("simple_daily_return", "mean"),
            benchmark_daily_return=("benchmark_daily_return", "mean"),
            short_overlay_daily_return=("short_overlay_daily_return", "mean"),
            strategy_net_position=("strategy_net_position", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )
    add_simple_curve_columns(portfolio, "simple_daily_return", "simple")
    add_strategy_alias_columns(portfolio)
    add_simple_curve_columns(
        portfolio,
        "benchmark_daily_return",
        "benchmark",
    )
    add_simple_curve_columns(
        portfolio,
        "short_overlay_daily_return",
        "short_overlay",
    )
    portfolio["excess_daily_return"] = (
        portfolio["simple_daily_return"] - portfolio["benchmark_daily_return"]
    )
    portfolio["excess_cumulative_return"] = (
        portfolio["simple_cumulative_return"]
        - portfolio["benchmark_cumulative_return"]
    )
    portfolio.insert(0, "symbol", "ALL_SYMBOLS_EQUAL_WEIGHT")
    return portfolio


def summarize_portfolio(portfolio, summaries, all_trades):
    simple_daily_return = portfolio["simple_daily_return"].fillna(0)
    benchmark_daily_return = portfolio["benchmark_daily_return"].fillna(0)
    short_overlay_daily_return = portfolio["short_overlay_daily_return"].fillna(0)
    summary = {
        "symbol": "ALL_SYMBOLS_EQUAL_WEIGHT",
        "factor_id": FACTOR_ID,
        "factor_name": FACTOR_NAME,
        "start_date": portfolio["date"].min(),
        "end_date": portfolio["date"].max(),
        "total_days": len(portfolio),
        "symbol_count": summaries["symbol"].nunique(),
        "mean_daily_symbol_count": portfolio["symbol_count"].mean(),
        "mean_holding_symbol_count": portfolio["holding_symbol_count"].mean(),
        "mean_reverse_short_symbol_count": portfolio[
            "holding_symbol_count"
        ].mean(),
        "mean_strategy_net_position": portfolio["strategy_net_position"].mean(),
        "benchmark_holding_ratio": 1.0,
        "entry_count": summaries["entry_count"].sum(),
        "exit_count": summaries["exit_count"].sum(),
        "simple_total_return": portfolio["simple_cumulative_return"].iloc[-1],
        "simple_annual_return": annualized_return(
            portfolio["simple_cumulative_return"].iloc[-1],
            len(portfolio),
        ),
        "simple_max_drawdown": portfolio["simple_drawdown"].min(),
        "simple_max_drawdown_ratio": portfolio["simple_drawdown_ratio"].min(),
        "mean_simple_daily_return": simple_daily_return.mean(),
        "std_simple_daily_return": simple_daily_return.std(ddof=1),
        "simple_sharpe": annualized_sharpe(simple_daily_return),
        "strategy_total_return": portfolio[
            "strategy_cumulative_return"
        ].iloc[-1],
        "strategy_annual_return": annualized_return(
            portfolio["strategy_cumulative_return"].iloc[-1],
            len(portfolio),
        ),
        "strategy_max_drawdown": portfolio["strategy_drawdown"].min(),
        "strategy_max_drawdown_ratio": portfolio[
            "strategy_drawdown_ratio"
        ].min(),
        "mean_strategy_daily_return": portfolio["strategy_daily_return"].mean(),
        "std_strategy_daily_return": portfolio["strategy_daily_return"].std(ddof=1),
        "strategy_sharpe": annualized_sharpe(portfolio["strategy_daily_return"]),
        "benchmark_total_return": portfolio[
            "benchmark_cumulative_return"
        ].iloc[-1],
        "benchmark_annual_return": annualized_return(
            portfolio["benchmark_cumulative_return"].iloc[-1],
            len(portfolio),
        ),
        "benchmark_max_drawdown": portfolio["benchmark_drawdown"].min(),
        "benchmark_max_drawdown_ratio": portfolio[
            "benchmark_drawdown_ratio"
        ].min(),
        "mean_benchmark_daily_return": benchmark_daily_return.mean(),
        "std_benchmark_daily_return": benchmark_daily_return.std(ddof=1),
        "benchmark_sharpe": annualized_sharpe(benchmark_daily_return),
        "benchmark_daily_win_rate": safe_ratio(
            int((benchmark_daily_return > 0).sum()),
            len(benchmark_daily_return),
        ),
        "best_benchmark_daily_return": benchmark_daily_return.max(),
        "worst_benchmark_daily_return": benchmark_daily_return.min(),
        "short_overlay_total_return": portfolio[
            "short_overlay_cumulative_return"
        ].iloc[-1],
        "short_overlay_annual_return": annualized_return(
            portfolio["short_overlay_cumulative_return"].iloc[-1],
            len(portfolio),
        ),
        "mean_short_overlay_daily_return": short_overlay_daily_return.mean(),
        "std_short_overlay_daily_return": short_overlay_daily_return.std(ddof=1),
        "short_overlay_sharpe": annualized_sharpe(short_overlay_daily_return),
        "excess_total_return": portfolio[
            "excess_cumulative_return"
        ].iloc[-1],
        "excess_annual_return": annualized_return(
            portfolio["excess_cumulative_return"].iloc[-1],
            len(portfolio),
        ),
        "mean_excess_daily_return": portfolio["excess_daily_return"].mean(),
        "best_simple_daily_return": simple_daily_return.max(),
        "worst_simple_daily_return": simple_daily_return.min(),
    }

    closed_trades = all_trades[all_trades["status"] == "closed"]
    summary["trade_count"] = len(all_trades)
    summary["closed_trade_count"] = len(closed_trades)
    summary["open_trade_count"] = int((all_trades["status"] == "open").sum())
    if closed_trades.empty:
        summary["trade_win_rate"] = np.nan
        summary["mean_trade_simple_return"] = np.nan
        summary["median_trade_simple_return"] = np.nan
        summary["best_trade_simple_return"] = np.nan
        summary["worst_trade_simple_return"] = np.nan
        summary["mean_trade_benchmark_return"] = np.nan
        summary["mean_trade_short_overlay_return"] = np.nan
        summary["mean_trade_excess_return"] = np.nan
        summary["mean_holding_days"] = np.nan
    else:
        trade_returns = closed_trades["trade_simple_return"]
        summary["trade_win_rate"] = safe_ratio(
            int((trade_returns > 0).sum()),
            len(trade_returns),
        )
        summary["mean_trade_simple_return"] = trade_returns.mean()
        summary["median_trade_simple_return"] = trade_returns.median()
        summary["best_trade_simple_return"] = trade_returns.max()
        summary["worst_trade_simple_return"] = trade_returns.min()
        summary["mean_trade_benchmark_return"] = closed_trades[
            "trade_benchmark_return"
        ].mean()
        summary["mean_trade_short_overlay_return"] = closed_trades[
            "trade_short_overlay_return"
        ].mean()
        summary["mean_trade_excess_return"] = closed_trades[
            "trade_excess_return"
        ].mean()
        summary["mean_holding_days"] = closed_trades["holding_days"].mean()

    return summary


def filter_signal_summaries(summaries):
    signal_days = pd.to_numeric(
        summaries["signal_days"],
        errors="coerce",
    ).fillna(0)
    return summaries[signal_days > 0].copy()


def build_signal_plot_data(all_daily, summaries, trades):
    plot_summaries = filter_signal_summaries(summaries)
    signal_symbols = set(plot_summaries["symbol"])
    plot_daily = [
        daily for daily in all_daily if daily["symbol"].iloc[0] in signal_symbols
    ]
    plot_trades = trades[trades["symbol"].isin(signal_symbols)].copy()
    plot_portfolio = build_portfolio_daily(plot_daily) if plot_daily else None
    return plot_summaries, plot_trades, plot_portfolio


def save_combined_outputs(
    portfolio,
    summaries,
    trades,
    output_dir,
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

    all_summary_path = summary_dir / f"all_symbols_{FACTOR_LABEL}_simple_summary.csv"
    all_trade_path = trade_dir / f"all_symbols_{FACTOR_LABEL}_simple_trades.csv"
    portfolio_path = (
        portfolio_dir
        / f"all_symbols_{FACTOR_LABEL}_simple_equal_weight_daily.csv"
    )
    portfolio_figure_path = (
        figure_dir
        / f"all_symbols_{FACTOR_LABEL}_simple_equal_weight_equity.png"
    )
    return_summary_figure_path = (
        figure_dir / f"all_symbols_{FACTOR_LABEL}_simple_return_summary.png"
    )

    portfolio_summary = summarize_portfolio(portfolio, summaries, trades)
    all_summaries = pd.concat(
        [summaries, pd.DataFrame([portfolio_summary])],
        ignore_index=True,
        sort=False,
    )
    all_summaries.to_csv(all_summary_path, index=False)
    trades.to_csv(all_trade_path, index=False)
    portfolio.to_csv(portfolio_path, index=False)

    if plot_portfolio is None:
        plot_empty(
            portfolio_figure_path,
            f"Signal symbols equal weight {FACTOR_LABEL}",
        )
        plot_return_summary(plot_summaries, return_summary_figure_path)
    else:
        plot_portfolio_summary = summarize_portfolio(
            plot_portfolio,
            plot_summaries,
            plot_trades,
        )
        plot_all_summaries = pd.concat(
            [plot_summaries, pd.DataFrame([plot_portfolio_summary])],
            ignore_index=True,
            sort=False,
        )
        plot_simple_equity(
            plot_portfolio,
            portfolio_figure_path,
            f"Signal symbols equal weight {FACTOR_LABEL}",
        )
        plot_return_summary(plot_all_summaries, return_summary_figure_path)

    return {
        "all_summary_path": all_summary_path,
        "all_trade_path": all_trade_path,
        "portfolio_path": portfolio_path,
        "portfolio_figure_path": portfolio_figure_path,
        "return_summary_figure_path": return_summary_figure_path,
    }


def run_backtest(file_info, output_dir):
    symbol = file_info["symbol"]
    daily = load_factor_daily(file_info)
    trades = build_trade_table(daily, symbol)
    summary = summarize_daily(daily, symbol)
    summary = add_trade_summary(summary, trades)
    output_paths = save_symbol_outputs(daily, trades, summary, output_dir, symbol)
    return daily, trades, summary, output_paths


def main():
    args = parse_args()
    factor_output_dir = args.factor_output_dir.resolve()
    output_dir = args.output_dir.resolve()
    factor_files = discover_factor_files(factor_output_dir, args.symbol)
    failures = []
    all_daily = []
    all_trades = []
    summaries = []

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Factor output directory: {factor_output_dir}", flush=True)
    print(f"Backtest output directory: {output_dir}", flush=True)
    print(
        f"Backtest date window: {BACKTEST_START_DATE.date()} to "
        f"{BACKTEST_END_DATE.date()}",
        flush=True,
    )
    print(f"Symbols to backtest: {len(factor_files)}", flush=True)
    print(
        "Symbol list: "
        + ",".join(file_info["symbol"] for file_info in factor_files),
        flush=True,
    )

    for file_info in factor_files:
        symbol = file_info["symbol"]
        print(f"\n###### Backtest symbol: {symbol} ######", flush=True)
        try:
            daily, trades, summary, output_paths = run_backtest(
                file_info,
                output_dir,
            )
            all_daily.append(daily)
            all_trades.append(trades)
            summaries.append(summary)
            print(
                f"[{symbol}] strategy total return: "
                f"{summary['strategy_total_return']:.6f}; "
                f"benchmark: {summary['benchmark_total_return']:.6f}; "
                f"excess: {summary['excess_total_return']:.6f}",
                flush=True,
            )
            print(f"[{symbol}] daily table: {output_paths['daily_path']}", flush=True)
            print(f"[{symbol}] trade table: {output_paths['trade_path']}", flush=True)
            print(
                f"[{symbol}] summary table: {output_paths['summary_path']}",
                flush=True,
            )
        except Exception as exc:
            if not args.keep_going:
                raise
            failures.append((symbol, exc))
            print(f"[{symbol}] backtest failed: {exc}", flush=True)

    if not summaries:
        raise RuntimeError("No symbols were backtested successfully.")

    summary_table = pd.DataFrame(summaries).sort_values("symbol")
    non_empty_trades = [trades for trades in all_trades if not trades.empty]
    trade_table = (
        pd.concat(non_empty_trades, ignore_index=True, sort=False)
        if non_empty_trades
        else all_trades[0].copy()
    )
    portfolio = build_portfolio_daily(all_daily)
    plot_summaries, plot_trades, plot_portfolio = build_signal_plot_data(
        all_daily,
        summary_table,
        trade_table,
    )
    combined_paths = save_combined_outputs(
        portfolio=portfolio,
        summaries=summary_table,
        trades=trade_table,
        output_dir=output_dir,
        plot_summaries=plot_summaries,
        plot_trades=plot_trades,
        plot_portfolio=plot_portfolio,
    )

    print("\nBacktest complete.", flush=True)
    print(f"Successful symbols: {len(summaries)}", flush=True)
    print(f"All-symbol summary: {combined_paths['all_summary_path']}", flush=True)
    print(f"All-symbol trades: {combined_paths['all_trade_path']}", flush=True)
    print(f"Equal-weight daily: {combined_paths['portfolio_path']}", flush=True)
    print(
        f"Return summary figure: {combined_paths['return_summary_figure_path']}",
        flush=True,
    )

    if failures:
        print(f"\nFailed symbols: {len(failures)}", flush=True)
        for symbol, exc in failures:
            print(f"- {symbol}: {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
