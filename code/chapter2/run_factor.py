import argparse
import importlib
from pathlib import Path

import pandas as pd

from core import io
from core.paths import DAILY_DIR, HOURLY_DIR, factor_output_dir
from core.plots import (
    plot_all_symbols,
    plot_return_summary,
    plot_signal_price,
    plot_strategy_curve,
)
from core.reports import empty_trade_table, output_paths, save_tables
from engines import backtest
from engines.daily_engine import run_daily_engine
from engines.hourly_exit_engine import run_hourly_exit_engine


def parse_args():
    parser = argparse.ArgumentParser(description="运行 chapter2 因子。")
    parser.add_argument("--factor", choices=["21", "22", "23", "24"], required=True)
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--hourly-dir", type=Path, default=HOURLY_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--symbol", action="append", help="可重复传入，默认全部品种。")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def load_factor(factor_id):
    return importlib.import_module(f"factors.factor_{factor_id}")


def discover_symbols(factor, daily_dir, hourly_dir, selected):
    if factor.ENGINE == "daily":
        symbols = io.discover_daily_symbols(daily_dir)
    else:
        symbols = io.discover_common_symbols(daily_dir, hourly_dir)
    return io.select_symbols(symbols, selected)


def run_symbol(symbol, factor, daily_dir, hourly_dir):
    daily = io.load_daily(symbol, daily_dir)
    if factor.ENGINE == "daily":
        frame = run_daily_engine(
            daily,
            factor.ENTRY_CONFIG,
            factor.EXIT_CONFIG,
            factor.USE_SPECULATION,
        )
        is_hourly = False
    else:
        hourly = io.load_hourly(symbol, hourly_dir)
        frame = run_hourly_exit_engine(
            daily,
            hourly,
            factor.ENTRY_CONFIG,
            factor.EXIT_CONFIG,
            factor.USE_SPECULATION,
        )
        is_hourly = True

    frame = backtest.add_return_columns(frame)
    periods = backtest.infer_periods_per_year(frame, is_hourly)
    trades = backtest.build_trade_table(
        frame, symbol, factor.FACTOR_ID, factor.FACTOR_NAME
    )
    metric = backtest.summarize_symbol(
        frame,
        trades,
        symbol,
        factor.FACTOR_ID,
        factor.FACTOR_NAME,
        periods,
    )
    curve = backtest.build_curve_table(
        frame, symbol, factor.FACTOR_ID, factor.FACTOR_NAME
    )
    return frame, trades, metric, curve, periods


def plot_symbol_outputs(symbol, frame, trades, curve, paths, factor):
    plot_signal_price(
        frame,
        trades,
        paths["signals"] / f"{symbol}_signals.png",
        f"{symbol} factor {factor.FACTOR_ID} signals",
    )
    plot_strategy_curve(
        curve,
        paths["strategy"] / f"{symbol}_strategy.png",
        f"{symbol} factor {factor.FACTOR_ID} strategy",
    )


def combine_outputs(symbol_results, factor):
    metrics = pd.DataFrame([item["metric"] for item in symbol_results])
    trade_tables = [item["trades"] for item in symbol_results if not item["trades"].empty]
    trades = (
        pd.concat(trade_tables, ignore_index=True)
        if trade_tables
        else empty_trade_table()
    )
    curves = pd.concat([item["curve"] for item in symbol_results], ignore_index=True)
    periods = max(item["periods"] for item in symbol_results)
    portfolio, portfolio_metric = backtest.build_portfolio(
        curves, factor.FACTOR_ID, factor.FACTOR_NAME, periods
    )
    metrics = pd.concat([metrics, pd.DataFrame([portfolio_metric])], ignore_index=True)
    return metrics, trades, curves, portfolio


def plot_combined_outputs(output_dir, metrics, curves, portfolio, factor):
    paths = output_paths(output_dir)
    plot_all_symbols(
        portfolio,
        paths["summary"],
        f"factor {factor.FACTOR_ID} all symbols summary",
    )
    plot_return_summary(
        curves,
        metrics,
        paths["return_summary"],
        f"{factor.FACTOR_ID}_{factor.FACTOR_NAME} return summary",
    )


def main():
    args = parse_args()
    factor = load_factor(args.factor)
    output_dir = (args.output_dir or factor_output_dir(factor.FACTOR_ID)).resolve()
    paths = output_paths(output_dir)
    symbols = discover_symbols(factor, args.daily_dir, args.hourly_dir, args.symbol)
    failures = []
    results = []

    print(f"运行因子：{factor.FACTOR_ID} {factor.FACTOR_NAME}", flush=True)
    print(f"输出目录：{output_dir}", flush=True)

    for symbol in symbols:
        try:
            frame, trades, metric, curve, periods = run_symbol(
                symbol, factor, args.daily_dir, args.hourly_dir
            )
            plot_symbol_outputs(symbol, frame, trades, curve, paths, factor)
            results.append(
                {
                    "symbol": symbol,
                    "trades": trades,
                    "metric": metric,
                    "curve": curve,
                    "periods": periods,
                }
            )
            print(f"[{symbol}] 完成，交易数：{metric['trade_count']}", flush=True)
        except Exception as exc:
            if not args.keep_going:
                raise
            failures.append((symbol, exc))
            print(f"[{symbol}] 失败：{exc}", flush=True)

    if not results:
        raise RuntimeError("没有任何品种成功完成")

    metrics, trades, curves, portfolio = combine_outputs(results, factor)
    save_tables(output_dir, metrics, trades, curves, portfolio)
    plot_combined_outputs(output_dir, metrics, curves, portfolio, factor)

    print(f"汇总表：{paths['metrics']}", flush=True)
    print(f"交易表：{paths['trades']}", flush=True)
    print(f"收益对比图：{paths['return_summary']}", flush=True)
    print(f"全品种图：{paths['summary']}", flush=True)

    if failures:
        for symbol, exc in failures:
            print(f"- {symbol}: {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
