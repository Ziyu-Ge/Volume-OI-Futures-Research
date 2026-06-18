import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from dow_theory_cta import (  # noqa: E402
    ANNUAL_DAYS,
    M,
    M_GRID,
    X_GRID,
    initial_position,
    normalize_dates_for_output,
    parse_int_list,
    run_best_from_comparison,
    run_dow_theory_cta,
    run_parameter_sweep,
    setup_matplotlib,
    x,
)


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
    "win_rate",
    "psychological_win_rate",
    "daily_win_rate",
    "traditional_win_rate",
    "num_trading_days",
    "num_long_days",
    "num_short_days",
    "num_flat_days",
    "num_trades",
    "num_completed_trades",
    "final_nav",
    "benchmark_final_nav",
    "param_compare_file",
    "best_daily_file",
    "best_summary_file",
    "best_nav_figure",
    "best_channels_figure",
    "single_daily_file",
    "single_summary_file",
    "single_nav_figure",
    "single_channels_figure",
]

ERROR_COLUMNS = ["symbol", "data_path", "error"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run dow_theory_cta.py's full default output flow for every symbol "
            "in data/: parameter comparison plus best-parameter detail files."
        )
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data"),
        help="Folder containing symbol CSV files. Defaults to project data/.",
    )
    parser.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT / "results" / "tables" / "dow_theory_cta_full_all_symbols"
        ),
        help="Folder for all-symbol tables and per-symbol detail CSV files.",
    )
    parser.add_argument(
        "--figure-root",
        default=str(
            PROJECT_ROOT / "results" / "figures" / "dow_theory_cta_full_all_symbols"
        ),
        help="Folder for all-symbol figures and per-symbol detail figures.",
    )
    parser.add_argument(
        "--M-list",
        default=",".join(str(value) for value in M_GRID),
        help="Comma-separated M values to scan.",
    )
    parser.add_argument(
        "--x-list",
        default=",".join(str(value) for value in X_GRID),
        help="Comma-separated x values to scan.",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Optional comma-separated symbols. Defaults to all CSV files in data-root.",
    )
    parser.add_argument(
        "--initial-position",
        type=int,
        choices=[-1, 0, 1],
        default=initial_position,
        help="Initial position before the first signal.",
    )
    parser.add_argument(
        "--include-single",
        action="store_true",
        help=(
            "Also run dow_theory_cta.py's --single style fixed M/x outputs "
            "under single/backtest, single/summary, and single figures."
        ),
    )
    parser.add_argument(
        "--single-M",
        type=int,
        default=M,
        help="M used when --include-single is set.",
    )
    parser.add_argument(
        "--single-x",
        type=int,
        default=x,
        help="x used when --include-single is set.",
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


def add_empty_single_paths(record):
    record["single_daily_file"] = ""
    record["single_summary_file"] = ""
    record["single_nav_figure"] = ""
    record["single_channels_figure"] = ""
    return record


def run_single_outputs(
    symbol,
    data_path,
    output_root,
    figure_root,
    single_M,
    single_x,
    initial_position_value,
):
    _, _, output_paths = run_dow_theory_cta(
        data_path=data_path,
        output_root=Path(output_root) / "single",
        figure_root=Path(figure_root) / "single",
        M_value=single_M,
        x_value=single_x,
        initial_position_value=initial_position_value,
        symbol=symbol,
        symbol_prefix=True,
    )
    return output_paths


def run_one_symbol(
    symbol,
    data_path,
    output_root,
    figure_root,
    M_values,
    x_values,
    initial_position_value,
    include_single,
    single_M,
    single_x,
):
    comparison, compare_path = run_parameter_sweep(
        data_path=data_path,
        output_root=output_root,
        M_values=M_values,
        x_values=x_values,
        initial_position_value=initial_position_value,
        symbol=symbol,
    )
    _, best_summary, best_output_paths, best_M, best_x = run_best_from_comparison(
        comparison=comparison,
        data_path=data_path,
        output_root=output_root,
        figure_root=figure_root,
        initial_position_value=initial_position_value,
        symbol=symbol,
    )

    record = best_summary.iloc[0].copy()
    record["rank"] = np.nan
    record["symbol"] = symbol
    record["M"] = best_M
    record["x"] = best_x
    record["benchmark_annual_return"] = annualize_return(
        record["benchmark_final_nav"],
        int(record["num_trading_days"]),
    )
    record["excess_annual_return"] = (
        record["annual_return"] - record["benchmark_annual_return"]
    )
    record["param_compare_file"] = str(compare_path)
    record["best_daily_file"] = str(best_output_paths["daily"])
    record["best_summary_file"] = str(best_output_paths["summary"])
    record["best_nav_figure"] = str(best_output_paths["nav"])
    record["best_channels_figure"] = str(best_output_paths["channels"])
    add_empty_single_paths(record)

    if include_single:
        single_paths = run_single_outputs(
            symbol=symbol,
            data_path=data_path,
            output_root=output_root,
            figure_root=figure_root,
            single_M=single_M,
            single_x=single_x,
            initial_position_value=initial_position_value,
        )
        record["single_daily_file"] = str(single_paths["daily"])
        record["single_summary_file"] = str(single_paths["summary"])
        record["single_nav_figure"] = str(single_paths["nav"])
        record["single_channels_figure"] = str(single_paths["channels"])

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


def save_all_symbol_summary(summary, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "all_symbols_full_summary.csv"
    normalize_dates_for_output(summary).to_csv(output_path, index=False)
    return output_path


def save_errors(errors, output_root):
    if not errors:
        return None

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "all_symbols_errors.csv"
    pd.DataFrame(errors, columns=ERROR_COLUMNS).to_csv(output_path, index=False)
    return output_path


def plot_annual_return_comparison(summary, figure_root):
    figure_root = Path(figure_root)
    figure_root.mkdir(parents=True, exist_ok=True)
    output_path = figure_root / "all_symbols_best_annual_return.png"

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
        label="Dow Theory CTA Best",
        color="#1f77b4",
        alpha=0.9,
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["symbol"])
    ax.set_xlabel("Annual Return (%)")
    ax.set_title("Dow Theory CTA Best Annual Return by Symbol")
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
    M_values = parse_int_list(args.M_list)
    x_values = parse_int_list(args.x_list)
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
                M_values=M_values,
                x_values=x_values,
                initial_position_value=args.initial_position,
                include_single=args.include_single,
                single_M=args.single_M,
                single_x=args.single_x,
            )
            records.append(record)
            print(
                "  best: "
                f"M={int(record['M'])}, x={int(record['x'])}, "
                f"annual_return={record['annual_return']:.2%}, "
                f"sharpe={record['sharpe_ratio']:.2f}"
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

    print("All-symbol Dow Theory CTA full run complete.")
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
