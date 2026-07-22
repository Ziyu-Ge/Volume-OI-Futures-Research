import argparse
from pathlib import Path

from core import io
from core.paths import DAILY_DIR, HOURLY_DIR, RESULTS_DIR
from core.plots import plot_all_symbols, plot_return_summary
from core.reports import output_paths, save_tables
from rolling.factor_24_param_space import build_param_candidates
from rolling.walk_forward import (
    FACTOR_ID,
    FACTOR_NAME,
    load_symbol_data,
    run_walk_forward,
)


def parse_args():
    parser = argparse.ArgumentParser(description="运行 24 号因子的滚动调参回测。")
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--hourly-dir", type=Path, default=HOURLY_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "factor_24_rolling",
    )
    parser.add_argument("--symbol", action="append", help="可重复传入，默认全部品种。")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument(
        "--drawdown-penalty",
        type=float,
        default=2.0,
        help="训练期打分：sharpe - drawdown_penalty * abs(max_drawdown)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    paths = output_paths(output_dir)

    symbols = io.discover_common_symbols(args.daily_dir, args.hourly_dir)
    symbols = io.select_symbols(symbols, args.symbol)
    candidates = build_param_candidates()

    print("运行因子：24 rolling walk-forward", flush=True)
    print(f"训练窗口：{args.train_years} 年", flush=True)
    print(f"调参频率/样本外窗口：{args.test_months} 个月", flush=True)
    print(f"候选参数组数：{len(candidates)}", flush=True)
    print(f"品种数量：{len(symbols)}", flush=True)
    print(f"输出目录：{output_dir}", flush=True)

    symbol_data = load_symbol_data(symbols, args.daily_dir, args.hourly_dir)
    result = run_walk_forward(
        symbol_data,
        candidates,
        train_years=args.train_years,
        test_months=args.test_months,
        drawdown_penalty=args.drawdown_penalty,
    )

    save_tables(
        output_dir,
        result["metrics"],
        result["trades"],
        result["curves"],
        result["portfolio"],
    )
    io.write_csv(
        result["selected_params"],
        output_dir / "tables" / "selected_params.csv",
    )

    plot_all_symbols(
        result["portfolio"],
        paths["summary"],
        f"{FACTOR_ID} {FACTOR_NAME} all symbols OOS summary",
    )
    plot_return_summary(
        result["curves"],
        result["metrics"],
        paths["return_summary"],
        f"{FACTOR_ID}_{FACTOR_NAME} OOS return summary",
    )

    portfolio_row = result["metrics"].loc[
        result["metrics"]["symbol"] == "ALL_SYMBOLS_EQUAL_WEIGHT"
    ].iloc[0]
    print("样本外组合指标：", flush=True)
    print(f"annual_return={portfolio_row['annual_return']:.6f}", flush=True)
    print(f"max_drawdown={portfolio_row['max_drawdown']:.6f}", flush=True)
    print(f"sharpe={portfolio_row['sharpe']:.6f}", flush=True)
    print(f"参数选择表：{output_dir / 'tables' / 'selected_params.csv'}", flush=True)
    print(f"汇总表：{paths['metrics']}", flush=True)
    print(f"交易表：{paths['trades']}", flush=True)
    print(f"收益对比图：{paths['return_summary']}", flush=True)
    print(f"全品种图：{paths['summary']}", flush=True)


if __name__ == "__main__":
    main()
