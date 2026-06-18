import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cpv_strategy import (  # noqa: E402
    ANNUAL_DAYS,
    run_cpv_strategy,
)


ALL_SYMBOL_COLUMNS = [
    "rank",
    "symbol",
    "start_date",
    "end_date",
    "annual_return",
    "benchmark_annual_return",
    "excess_annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "num_backtest_days",
    "num_trading_days",
    "num_long_days",
    "num_short_days",
    "num_flat_days",
    "num_turnovers",
    "final_nav",
    "benchmark_final_nav",
    "factor_file",
    "backtest_file",
    "summary_file",
    "simple_nav_figure",
    "compound_nav_figure",
]

ERROR_COLUMNS = ["symbol", "data_path", "error"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the CPV strategy for every symbol CSV in data/."
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data"),
        help="Folder containing symbol CSV files. Defaults to project data/.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "results" / "tables" / "cpv_all_symbols"),
        help="Folder for all-symbol CPV output tables.",
    )
    parser.add_argument(
        "--figure-root",
        default=str(PROJECT_ROOT / "results" / "figures" / "cpv_all_symbols"),
        help="Folder for all-symbol CPV NAV figures.",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Optional comma-separated symbols. Defaults to all CSV files in data-root.",
    )
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=30,
        help="Minimum valid minute deltas required for one daily PV value.",
    )
    parser.add_argument(
        "--cost-rate",
        type=float,
        default=0.0,
        help="Trading cost charged for one unit of position change.",
    )
    return parser.parse_args()


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
    ]


def annualize_return(final_nav, num_days):
    if pd.notna(final_nav) and final_nav > 0 and num_days > 0:
        return final_nav ** (ANNUAL_DAYS / num_days) - 1
    return np.nan


def run_one_symbol(
    symbol,
    data_path,
    output_root,
    figure_root,
    min_minutes,
    cost_rate,
):
    _, backtest, summary, output_paths = run_cpv_strategy(
        symbol=symbol,
        data_path=data_path,
        output_root=output_root,
        figure_root=figure_root,
        min_minutes=min_minutes,
        cost_rate=cost_rate,
    )

    record = summary.iloc[0].copy()
    num_backtest_days = len(backtest)
    final_nav = (
        backtest["strategy_nav"].iloc[-1]
        if num_backtest_days > 0
        else np.nan
    )
    benchmark_final_nav = (
        backtest["benchmark_nav"].iloc[-1]
        if num_backtest_days > 0
        else np.nan
    )
    benchmark_annual_return = annualize_return(
        benchmark_final_nav,
        num_backtest_days,
    )

    record["rank"] = np.nan
    record["num_backtest_days"] = num_backtest_days
    record["final_nav"] = final_nav
    record["benchmark_final_nav"] = benchmark_final_nav
    record["benchmark_annual_return"] = benchmark_annual_return
    record["excess_annual_return"] = (
        record["annual_return"] - benchmark_annual_return
    )
    record["factor_file"] = str(output_paths["factor"])
    record["backtest_file"] = str(output_paths["backtest"])
    record["summary_file"] = str(output_paths["summary"])
    record["simple_nav_figure"] = str(output_paths["simple_nav"])
    record["compound_nav_figure"] = str(output_paths["compound_nav"])

    return record


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
    out = df.copy()
    for column in ["start_date", "end_date"]:
        out[column] = pd.to_datetime(
            out[column],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
    return out


def save_all_symbol_summary(summary, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "all_symbols_summary.csv"
    format_date_columns(summary).to_csv(output_path, index=False)
    return output_path


def save_errors(errors, output_root):
    if not errors:
        return None

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "all_symbols_errors.csv"
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


def plot_annual_return_comparison(summary, figure_root):
    figure_root = Path(figure_root)
    figure_root.mkdir(parents=True, exist_ok=True)
    output_path = figure_root / "all_symbols_annual_return.png"

    plt = setup_matplotlib()
    plot_df = summary.sort_values("annual_return", ascending=True).copy()
    y_pos = np.arange(len(plot_df))
    bar_height = 0.38
    figure_height = max(8, len(plot_df) * 0.32 + 2)

    fig, ax = plt.subplots(figsize=(13, figure_height))
    ax.barh(
        y_pos - bar_height / 2,
        plot_df["benchmark_annual_return"] * 100,
        height=bar_height,
        label="Benchmark",
        color="#8c8c8c",
        alpha=0.8,
    )
    ax.barh(
        y_pos + bar_height / 2,
        plot_df["annual_return"] * 100,
        height=bar_height,
        label="CPV Strategy",
        color="#2ca02c",
        alpha=0.9,
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["symbol"])
    ax.set_xlabel("Annual Return (%)")
    ax.set_title("CPV Strategy Annual Return by Symbol")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    figure_root = Path(args.figure_root)
    symbol_paths = discover_symbol_paths(data_root, args.symbols)

    if not symbol_paths:
        raise ValueError(f"No CSV files found in {data_root}")

    records = []
    errors = []

    for index, (symbol, data_path) in enumerate(symbol_paths, start=1):
        print(f"[{index}/{len(symbol_paths)}] Running {symbol} from {data_path}")
        try:
            if not data_path.exists():
                raise FileNotFoundError(data_path)

            record = run_one_symbol(
                symbol=symbol,
                data_path=data_path,
                output_root=output_root,
                figure_root=figure_root,
                min_minutes=args.min_minutes,
                cost_rate=args.cost_rate,
            )
            records.append(record)
            print(
                "  done: "
                f"annual_return={record['annual_return']:.2%}, "
                f"sharpe={record['sharpe_ratio']:.2f}, "
                f"final_nav={record['final_nav']:.4f}"
            )
        except Exception as exc:
            errors.append({
                "symbol": symbol,
                "data_path": str(data_path),
                "error": repr(exc),
            })
            print(f"  failed: {exc}")

    if not records:
        error_path = save_errors(errors, output_root)
        raise RuntimeError(f"No symbols completed successfully. errors={error_path}")

    summary = build_all_symbol_summary(records)
    summary_path = save_all_symbol_summary(summary, output_root)
    figure_path = plot_annual_return_comparison(summary, figure_root)
    error_path = save_errors(errors, output_root)

    print("All-symbol CPV strategy run complete.")
    print(f"symbols completed: {len(summary)}")
    print(f"symbols failed: {len(errors)}")
    print(f"summary file: {summary_path}")
    print(f"comparison figure: {figure_path}")
    if error_path is not None:
        print(f"error file: {error_path}")
    print(
        summary.loc[
            :,
            [
                "rank",
                "symbol",
                "annual_return",
                "benchmark_annual_return",
                "sharpe_ratio",
                "max_drawdown",
                "final_nav",
            ],
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
