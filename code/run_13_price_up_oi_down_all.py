import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

FACTOR_ID = "13"
FACTOR_NAME = "price_up_oi_down"
FACTOR_SCRIPT = CODE_DIR / "factors" / "13price_up_oi_down.py"
PREPARE_SCRIPT = CODE_DIR / "01_prepare_data.py"
BACKTEST_SCRIPT = CODE_DIR / "backtest" / "backtest_sum.py"
DEFAULT_OUTPUT_DIR = (
    RESULTS_DIR / "runs" / "13_price_up_oi_down_15min_all_symbols"
)
FACTOR_DATA_FREQUENCY = "15min"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

RUN_SCRIPT_CODE = textwrap.dedent(
    """
    import importlib.util
    import os
    import runpy
    import sys

    project_root, code_dir, script_path, symbol = sys.argv[1:5]

    os.chdir(project_root)
    os.environ["SYMBOL"] = symbol

    for path in [os.path.dirname(script_path), code_dir, project_root]:
        if path and path not in sys.path:
            sys.path.insert(0, path)

    config_path = os.path.join(code_dir, "config.py")
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config.SYMBOL = symbol
    sys.modules["config"] = config

    runpy.run_path(script_path, run_name="__main__")
    """
)


def parse_symbols(raw_value):
    if raw_value is None:
        return None

    symbols = [
        item.strip().upper()
        for item in raw_value.split(",")
        if item.strip()
    ]

    return symbols or None


def discover_symbols(symbol_filter=None):
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"数据目录不存在：{DATA_DIR}")

    available_symbols = sorted(
        path.stem.upper()
        for path in DATA_DIR.glob("*.csv")
        if path.is_file()
    )
    if not available_symbols:
        raise FileNotFoundError(f"数据目录中没有 CSV 文件：{DATA_DIR}")

    if symbol_filter is None:
        return available_symbols

    available_set = set(available_symbols)
    missing_symbols = [
        symbol
        for symbol in symbol_filter
        if symbol not in available_set
    ]
    if missing_symbols:
        raise FileNotFoundError(
            "以下品种在 data/ 中没有找到对应 CSV："
            f"{','.join(missing_symbols)}"
        )

    return symbol_filter


def build_env(symbol):
    env = os.environ.copy()
    env["SYMBOL"] = symbol
    env["FACTOR_ID"] = FACTOR_ID
    env["FACTOR_DATA_FREQUENCY"] = FACTOR_DATA_FREQUENCY
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
    return env


def run_script(script_path, symbol, env):
    relative_path = script_path.relative_to(PROJECT_ROOT)
    print(f"\n========== [{symbol}] run {relative_path} ==========", flush=True)

    subprocess.run(
        [
            sys.executable,
            "-c",
            RUN_SCRIPT_CODE,
            str(PROJECT_ROOT),
            str(CODE_DIR),
            str(script_path),
            symbol,
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def collect_file(source_path, destination_path):
    if not source_path.exists():
        return False

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return True


def collect_symbol_outputs(symbol, output_dir, include_backtest=True):
    filename_prefix = f"{symbol}_{FACTOR_ID}_{FACTOR_NAME}"
    source_files = [
        (
            RESULTS_DIR / "tables" / "15min" / f"{symbol}_15min.csv",
            output_dir / "tables" / "15min" / f"{symbol}_15min.csv",
        ),
        (
            RESULTS_DIR / "tables" / "factors" / f"{filename_prefix}.csv",
            output_dir / "tables" / "factors" / f"{filename_prefix}.csv",
        ),
        (
            RESULTS_DIR / "tables" / "signals" / f"{filename_prefix}_signals.csv",
            output_dir / "tables" / "signals" / f"{filename_prefix}_signals.csv",
        ),
        (
            RESULTS_DIR / "tables" / "summary" / f"{filename_prefix}_summary.csv",
            output_dir / "tables" / "summary" / f"{filename_prefix}_summary.csv",
        ),
        (
            RESULTS_DIR
            / "figures"
            / f"{filename_prefix}_signal_on_price.png",
            output_dir
            / "figures"
            / "factors"
            / f"{filename_prefix}_signal_on_price.png",
        ),
        (
            RESULTS_DIR
            / "figures"
            / f"{filename_prefix}_factor_value.png",
            output_dir
            / "figures"
            / "factors"
            / f"{filename_prefix}_factor_value.png",
        ),
    ]

    if include_backtest:
        source_files.extend([
            (
                RESULTS_DIR
                / "tables"
                / "backtest"
                / f"{filename_prefix}_long_only_simple_backtest.csv",
                output_dir
                / "tables"
                / "backtest"
                / f"{filename_prefix}_long_only_simple_backtest.csv",
            ),
            (
                RESULTS_DIR
                / "tables"
                / "backtest"
                / f"{filename_prefix}_long_only_simple_summary.csv",
                output_dir
                / "tables"
                / "backtest"
                / f"{filename_prefix}_long_only_simple_summary.csv",
            ),
            (
                RESULTS_DIR
                / "figures"
                / "backtest"
                / f"{filename_prefix}_simple_nav.png",
                output_dir
                / "figures"
                / "backtest"
                / f"{filename_prefix}_simple_nav.png",
            ),
        ])

    missing_files = []
    for source_path, destination_path in source_files:
        if not collect_file(source_path, destination_path):
            missing_files.append(source_path)

    return missing_files


def read_backtest_summary(symbol, output_dir):
    summary_path = (
        output_dir
        / "tables"
        / "backtest"
        / f"{symbol}_{FACTOR_ID}_{FACTOR_NAME}_long_only_simple_summary.csv"
    )
    if not summary_path.exists():
        return None

    summary = pd.read_csv(summary_path)
    summary.insert(0, "symbol", symbol)
    return summary


def build_wide_summary(combined_summary):
    metric_columns = [
        "total_return",
        "annual_return",
        "max_drawdown",
        "annual_volatility",
        "sharpe",
        "signal_days",
        "signal_ratio",
    ]
    wide = combined_summary.pivot_table(
        index="symbol",
        columns="strategy",
        values=metric_columns,
        aggfunc="first",
    )
    wide.columns = [
        f"{strategy}_{metric}"
        for metric, strategy in wide.columns
    ]
    wide = wide.reset_index()

    base_prefix = "base_long_only_simple"
    strategy_prefix = "factor_position_simple"
    for metric in [
        "total_return",
        "annual_return",
        "max_drawdown",
        "annual_volatility",
        "sharpe",
    ]:
        base_col = f"{base_prefix}_{metric}"
        strategy_col = f"{strategy_prefix}_{metric}"
        if base_col not in wide.columns or strategy_col not in wide.columns:
            continue

        wide[f"excess_{metric}"] = wide[strategy_col] - wide[base_col]

    sort_col = "excess_total_return"
    if sort_col in wide.columns:
        wide = wide.sort_values(sort_col, ascending=False)
    else:
        wide = wide.sort_values("symbol")

    return wide.reset_index(drop=True)


def save_summary_figure(wide_summary, output_dir):
    strategy_return_col = "factor_position_simple_total_return"
    base_return_col = "base_long_only_simple_total_return"
    excess_return_col = "excess_total_return"

    required_columns = {
        "symbol",
        strategy_return_col,
        base_return_col,
        excess_return_col,
    }
    if not required_columns.issubset(wide_summary.columns):
        return None

    plot_data = wide_summary.sort_values(excess_return_col)
    y_positions = np.arange(len(plot_data))
    figure_height = max(8, len(plot_data) * 0.28 + 2)

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(16, figure_height),
        gridspec_kw={"width_ratios": [1.4, 1]},
    )

    axes[0].barh(
        y_positions - 0.18,
        plot_data[base_return_col],
        height=0.36,
        label="Base Long",
        color="#6b7280",
    )
    axes[0].barh(
        y_positions + 0.18,
        plot_data[strategy_return_col],
        height=0.36,
        label="Factor Strategy",
        color="#2563eb",
    )
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(plot_data["symbol"])
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_title("Total Return")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower right")

    excess_colors = np.where(
        plot_data[excess_return_col] >= 0,
        "#16a34a",
        "#dc2626",
    )
    axes[1].barh(
        y_positions,
        plot_data[excess_return_col],
        color=excess_colors,
    )
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels([])
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_title("Excess Total Return")
    axes[1].axvline(0, color="#111827", linewidth=0.8)
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"Factor {FACTOR_ID}: {FACTOR_NAME} All Symbols Backtest Summary",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    figure_path = (
        output_dir
        / "figures"
        / "summary"
        / f"all_symbols_{FACTOR_ID}_{FACTOR_NAME}_summary.png"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    return figure_path


def save_combined_summaries(symbols, output_dir, require_backtest=True):
    summaries = []
    for symbol in symbols:
        summary = read_backtest_summary(symbol, output_dir)
        if summary is not None:
            summaries.append(summary)

    if not summaries and require_backtest:
        raise FileNotFoundError("没有找到可汇总的 13 号因子回测 summary。")
    if not summaries:
        return None

    combined_summary = pd.concat(summaries, ignore_index=True)
    wide_summary = build_wide_summary(combined_summary)

    summary_dir = output_dir / "tables" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    combined_summary_path = (
        summary_dir
        / f"all_symbols_{FACTOR_ID}_{FACTOR_NAME}_backtest_summary_long.csv"
    )
    wide_summary_path = (
        summary_dir
        / f"all_symbols_{FACTOR_ID}_{FACTOR_NAME}_backtest_summary.csv"
    )
    combined_summary.to_csv(combined_summary_path, index=False)
    wide_summary.to_csv(wide_summary_path, index=False)
    figure_path = save_summary_figure(wide_summary, output_dir)

    return {
        "combined_summary_path": combined_summary_path,
        "wide_summary_path": wide_summary_path,
        "figure_path": figure_path,
        "summary_count": len(summaries),
    }


def run_symbol(symbol, args, output_dir):
    env = build_env(symbol)

    if not args.collect_only:
        if not args.skip_prepare:
            run_script(PREPARE_SCRIPT, symbol, env)
        if not args.skip_factor:
            run_script(FACTOR_SCRIPT, symbol, env)
        if not args.skip_backtest:
            run_script(BACKTEST_SCRIPT, symbol, env)

    missing_files = collect_symbol_outputs(
        symbol,
        output_dir,
        include_backtest=not args.skip_backtest,
    )
    if missing_files:
        print(f"\n[{symbol}] 以下输出文件未找到，已跳过复制：", flush=True)
        for missing_file in missing_files:
            print(f"- {missing_file}", flush=True)

    return missing_files


def main():
    parser = argparse.ArgumentParser(
        description="运行全部品种的 13 号因子回测，并输出独立汇总目录。"
    )
    parser.add_argument(
        "--symbols",
        "--symbol",
        dest="symbols",
        help="只运行指定品种，多个品种用逗号分隔，例如：PL,CU。默认运行 data/ 下全部品种。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"独立结果目录，默认：{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="不重新运行脚本，只基于现有 results/ 复制 13 号因子输出并生成汇总。",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="跳过 15min 数据准备。",
    )
    parser.add_argument(
        "--skip-factor",
        action="store_true",
        help="跳过 13 号因子计算。",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="跳过回测，只复制已有输出并汇总。",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="单个品种失败后继续运行后续品种，并在最后汇总失败列表。",
    )

    args = parser.parse_args()
    symbols = discover_symbols(parse_symbols(args.symbols))
    output_dir = args.output_dir.resolve()
    failures = []

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"本次运行品种数量：{len(symbols)}", flush=True)
    print(f"品种列表：{','.join(symbols)}", flush=True)
    print(f"独立结果目录：{output_dir}", flush=True)

    for symbol in symbols:
        print(f"\n###### 开始处理品种：{symbol} ######", flush=True)
        try:
            run_symbol(symbol, args, output_dir)
        except Exception as exc:
            if not args.keep_going:
                raise

            failures.append((symbol, exc))
            print(f"\n!!!!!! 品种 {symbol} 处理失败：{exc} !!!!!!", flush=True)

    summary_info = save_combined_summaries(
        symbols,
        output_dir,
        require_backtest=not args.skip_backtest,
    )

    print("\n全部任务完成。", flush=True)
    print(f"成功处理品种数量：{len(symbols) - len(failures)}", flush=True)
    if summary_info is not None:
        print(f"汇总品种数量：{summary_info['summary_count']}", flush=True)
        print(f"汇总表：{summary_info['wide_summary_path']}", flush=True)
        print(f"长表汇总：{summary_info['combined_summary_path']}", flush=True)
        if summary_info["figure_path"] is not None:
            print(f"汇总图：{summary_info['figure_path']}", flush=True)
    else:
        print("已跳过回测汇总。", flush=True)

    if failures:
        print(f"\n失败品种数量：{len(failures)}", flush=True)
        for symbol, exc in failures:
            print(f"- {symbol}: {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
