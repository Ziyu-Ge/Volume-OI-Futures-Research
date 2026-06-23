import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
STRATEGY_PATH = CODE_ROOT / "factors" / "23_dow_theory_cta.py"
RUN_NAME = "23_dow_theory_cta_all_symbols"
DEFAULT_M_LIST = "10,20,30,60,90,120,180,270,360"
DEFAULT_X_LIST = "1,2,3,5,10,20"

ALL_SYMBOL_COLUMNS = [
    "rank",
    "symbol",
    "start_date",
    "end_date",
    "M",
    "x",
    "annual_return",
    "benchmark_annual_return",
    "excess_annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "psychological_win_rate",
    "daily_win_rate",
    "traditional_win_rate",
    "num_trading_days",
    "num_trades",
    "num_completed_trades",
    "final_nav",
    "benchmark_final_nav",
    "param_compare_file",
    "best_daily_file",
    "best_summary_file",
    "best_nav_figure",
    "best_channels_figure",
]

ERROR_COLUMNS = ["symbol", "data_path", "error"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run code/factors/23_dow_theory_cta.py for all symbol CSVs."
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data"),
        help="Folder containing symbol CSV files. Defaults to project data/.",
    )
    parser.add_argument(
        "--run-root",
        default=str(PROJECT_ROOT / "results" / "chapter2_runs" / RUN_NAME),
        help="Root folder for all outputs from this batch run.",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Optional comma-separated symbols. Defaults to every CSV in data-root.",
    )
    parser.add_argument(
        "--M-list",
        default=DEFAULT_M_LIST,
        help="Comma-separated M values to scan.",
    )
    parser.add_argument(
        "--x-list",
        default=DEFAULT_X_LIST,
        help="Comma-separated x values to scan.",
    )
    parser.add_argument(
        "--initial-position",
        type=int,
        choices=[-1, 0, 1],
        default=0,
        help="Initial position before the first signal.",
    )
    return parser.parse_args()


def load_strategy_module():
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))

    spec = importlib.util.spec_from_file_location(
        "dow_theory_cta_23",
        STRATEGY_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy module from {STRATEGY_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_int_list(value):
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            values.append(int(item))

    if not values:
        raise ValueError("parameter list cannot be empty")

    return values


def discover_symbol_paths(data_root, symbols=None):
    data_root = Path(data_root)
    if symbols:
        requested_symbols = [
            item.strip().upper()
            for item in symbols.split(",")
            if item.strip()
        ]
        return [(symbol, data_root / f"{symbol}.csv") for symbol in requested_symbols]

    return [
        (path.stem.upper(), path)
        for path in sorted(data_root.glob("*.csv"))
        if not path.name.startswith(".")
    ]


def annualize_return(final_nav, num_days, annual_days):
    if pd.notna(final_nav) and final_nav > 0 and num_days > 0:
        return final_nav ** (annual_days / num_days) - 1
    return np.nan


def run_one_symbol(
    strategy_module,
    symbol,
    data_path,
    output_root,
    figure_root,
    M_values,
    x_values,
    initial_position_value,
):
    comparison, compare_path = strategy_module.run_parameter_sweep(
        data_path=data_path,
        output_root=output_root,
        M_values=M_values,
        x_values=x_values,
        initial_position_value=initial_position_value,
        symbol=symbol,
    )

    best = comparison.iloc[0].copy()
    best["rank"] = np.nan
    best["param_compare_file"] = str(compare_path)

    _, _, output_paths, best_M, best_x = strategy_module.run_best_from_comparison(
        comparison=comparison,
        data_path=data_path,
        output_root=output_root,
        figure_root=figure_root,
        initial_position_value=initial_position_value,
        symbol=symbol,
    )

    best["M"] = best_M
    best["x"] = best_x
    best["benchmark_annual_return"] = annualize_return(
        best["benchmark_final_nav"],
        int(best["num_trading_days"]),
        strategy_module.ANNUAL_DAYS,
    )
    best["excess_annual_return"] = (
        best["annual_return"] - best["benchmark_annual_return"]
    )
    best["best_daily_file"] = str(output_paths["daily"])
    best["best_summary_file"] = str(output_paths["summary"])
    best["best_nav_figure"] = str(output_paths["nav"])
    best["best_channels_figure"] = str(output_paths["channels"])

    return best


def build_all_symbol_summary(records):
    summary = pd.DataFrame(records)
    summary = summary.sort_values(
        ["annual_return", "sharpe_ratio", "final_nav"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary[ALL_SYMBOL_COLUMNS]


def format_date_columns(df):
    output = df.copy()
    for column in ["start_date", "end_date"]:
        output[column] = pd.to_datetime(
            output[column],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
    return output


def save_all_symbol_summary(summary, summary_root):
    summary_root = Path(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)
    output_path = summary_root / "all_symbols_23_dow_theory_cta_best_summary.csv"
    format_date_columns(summary).to_csv(output_path, index=False)
    return output_path


def save_errors(errors, summary_root):
    if not errors:
        return None

    summary_root = Path(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)
    output_path = summary_root / "all_symbols_23_dow_theory_cta_errors.csv"
    pd.DataFrame(errors, columns=ERROR_COLUMNS).to_csv(output_path, index=False)
    return output_path


def setup_matplotlib():
    mpl_config_dir = Path(tempfile.gettempdir()) / "lc_research_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_summary(summary, figure_summary_root):
    figure_summary_root = Path(figure_summary_root)
    figure_summary_root.mkdir(parents=True, exist_ok=True)
    output_path = figure_summary_root / "all_symbols_23_dow_theory_cta_best_summary.png"

    plt = setup_matplotlib()
    plot_df = summary.sort_values("annual_return", ascending=True).copy()
    y_pos = np.arange(len(plot_df))
    figure_height = max(9, len(plot_df) * 0.34 + 2)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, figure_height),
        sharey=True,
        gridspec_kw={"width_ratios": [1.4, 1.0, 1.0]},
    )

    bar_height = 0.36
    axes[0].barh(
        y_pos - bar_height / 2,
        plot_df["benchmark_annual_return"] * 100,
        height=bar_height,
        label="Benchmark",
        color="#8a8f98",
        alpha=0.8,
    )
    axes[0].barh(
        y_pos + bar_height / 2,
        plot_df["annual_return"] * 100,
        height=bar_height,
        label="Dow CTA Best",
        color="#2f7f68",
        alpha=0.95,
    )
    axes[0].set_title("Annual Return")
    axes[0].set_xlabel("%")
    axes[0].legend(loc="lower right")

    axes[1].barh(
        y_pos,
        plot_df["sharpe_ratio"],
        color="#496b9c",
        alpha=0.9,
    )
    axes[1].set_title("Sharpe Ratio")
    axes[1].set_xlabel("Ratio")

    axes[2].barh(
        y_pos,
        plot_df["max_drawdown"] * 100,
        color="#b45c4d",
        alpha=0.9,
    )
    axes[2].set_title("Max Drawdown")
    axes[2].set_xlabel("%")

    for ax in axes:
        ax.axvline(0, color="#30343b", linewidth=0.8)
        ax.grid(True, axis="x", linestyle="--", alpha=0.25)
        ax.set_yticks(y_pos)

    axes[0].set_yticklabels(plot_df["symbol"])
    axes[0].set_ylabel("Symbol")
    fig.suptitle("23 Dow Theory CTA Best Summary by Symbol", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    run_root = Path(args.run_root)
    table_root = run_root / "tables"
    figure_root = run_root / "figures"
    summary_root = table_root / "summary"
    figure_backtest_root = figure_root / "backtest"
    figure_summary_root = figure_root / "summary"

    M_values = parse_int_list(args.M_list)
    x_values = parse_int_list(args.x_list)
    symbol_paths = discover_symbol_paths(data_root, args.symbols)
    if not symbol_paths:
        raise ValueError(f"No CSV files found in {data_root}")

    strategy_module = load_strategy_module()
    records = []
    errors = []

    for index, (symbol, data_path) in enumerate(symbol_paths, start=1):
        print(f"[{index}/{len(symbol_paths)}] Running {symbol} from {data_path}")
        try:
            if not data_path.exists():
                raise FileNotFoundError(data_path)

            best = run_one_symbol(
                strategy_module=strategy_module,
                symbol=symbol,
                data_path=data_path,
                output_root=table_root,
                figure_root=figure_backtest_root,
                M_values=M_values,
                x_values=x_values,
                initial_position_value=args.initial_position,
            )
            records.append(best)
            print(
                "  best: "
                f"M={int(best['M'])}, x={int(best['x'])}, "
                f"annual_return={best['annual_return']:.2%}, "
                f"sharpe={best['sharpe_ratio']:.2f}, "
                f"final_nav={best['final_nav']:.4f}"
            )
        except Exception as exc:
            errors.append({
                "symbol": symbol,
                "data_path": str(data_path),
                "error": repr(exc),
            })
            print(f"  failed: {exc}")

    error_path = save_errors(errors, summary_root)
    if not records:
        raise RuntimeError(f"No symbols completed successfully. errors={error_path}")

    summary = build_all_symbol_summary(records)
    summary_path = save_all_symbol_summary(summary, summary_root)
    figure_path = plot_summary(summary, figure_summary_root)

    print("All-symbol 23 Dow Theory CTA run complete.")
    print(f"symbols completed: {len(summary)}")
    print(f"symbols failed: {len(errors)}")
    print(f"summary file: {summary_path}")
    print(f"summary figure: {figure_path}")
    if error_path is not None:
        print(f"error file: {error_path}")
    print(
        summary.loc[
            :,
            [
                "rank",
                "symbol",
                "M",
                "x",
                "annual_return",
                "benchmark_annual_return",
                "sharpe_ratio",
                "max_drawdown",
                "num_trades",
                "final_nav",
            ],
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
