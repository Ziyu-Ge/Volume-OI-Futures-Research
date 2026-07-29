import argparse
from pathlib import Path

# combined 信号图的命令行入口。
from plot_combined_signals_config import (
    DEFAULT_FACTOR_IDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUNS_DIR,
)
from plot_combined_signals_charts import save_outputs
from plot_combined_signals_data import (
    discover_factor_files,
    load_all_factor_data,
    parse_factor_ids,
)


def main():
    """生成 combined 信号统计表、静态图、交互图和 dashboard。"""
    parser = argparse.ArgumentParser(
        description="Plot close price with all factor signal points from results."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"directory containing numbered factor result folders, default: {DEFAULT_RUNS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory, default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--factor-ids",
        default=",".join(DEFAULT_FACTOR_IDS),
        help=(
            "comma-separated factor ids to plot, default: "
            f"{','.join(DEFAULT_FACTOR_IDS)}; use ALL to plot every factor result"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="figure dpi, default: 300",
    )
    args = parser.parse_args()
    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not runs_dir.is_dir():
        raise FileNotFoundError(f"factor results directory not found: {runs_dir}")

    factor_files = discover_factor_files(
        runs_dir=runs_dir,
        factor_id_filter=parse_factor_ids(args.factor_ids),
    )
    factor_frames, errors = load_all_factor_data(factor_files)
    stats_path, figure_paths, html_paths, dashboard_path = save_outputs(
        factor_frames=factor_frames,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    print("combined signal outputs complete.")
    print(f"factor files loaded: {len(factor_frames)}")
    print(f"static png figures saved: {len(figure_paths)}")
    print(f"interactive html figures saved: {len(html_paths)}")
    print(f"dashboard: {dashboard_path}")
    print(f"stats table: {stats_path}")
    print(f"figures dir: {output_dir / 'figures'}")

    if errors:
        print("\nskipped files:")
        for path, exc in errors:
            print(f"- {path}: {exc}")


if __name__ == "__main__":
    main()
