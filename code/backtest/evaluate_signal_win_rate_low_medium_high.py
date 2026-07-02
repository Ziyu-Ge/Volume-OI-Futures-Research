import argparse
from pathlib import Path

import pandas as pd

import evaluate_signal_win_rate as base


DEFAULT_OUTPUT_DIR = base.PROJECT_ROOT / "results" / "evaluation_low_medium_high"
CONFIDENCE_LEVELS = set(base.CONFIDENCE_LEVELS)


def add_common_arguments(parser):
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=base.DEFAULT_RUNS_DIR,
        help=f"因子结果根目录，默认：{base.DEFAULT_RUNS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"评估结果输出目录，默认：{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--factor-ids",
        default=base.DEFAULT_FACTOR_IDS,
        help=f"要评估的因子 ID，逗号分隔，默认：{base.DEFAULT_FACTOR_IDS}",
    )
    parser.add_argument(
        "--symbols",
        "--symbol",
        dest="symbols",
        help="只评估指定品种，多个品种用逗号分隔，例如：JD,CU。",
    )
    parser.add_argument(
        "--price-column",
        default="close",
        help="用于评估的价格字段，默认：close。",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=None,
        help=(
            "只评估一个观察窗口，例如：10。"
            "如果没有设置该参数或 --lookahead-days-list，默认评估 3,5,10。"
        ),
    )
    parser.add_argument(
        "--lookahead-days-list",
        help="一次评估多个观察窗口，逗号分隔，例如：3,5,10。",
    )
    parser.add_argument(
        "--volatility-window",
        type=int,
        default=base.DEFAULT_VOLATILITY_WINDOW,
        help="计算前 N 日平均绝对日收益率，默认：20。",
    )
    parser.add_argument(
        "--volatility-min-history-days",
        type=int,
        default=base.DEFAULT_VOLATILITY_MIN_HISTORY_DAYS,
        help="计算平均波动率时最少需要多少个历史日收益，默认：20。",
    )
    parser.add_argument(
        "--cluster-max-gap",
        type=int,
        default=base.DEFAULT_CLUSTER_MAX_GAP,
        help=(
            "两个信号之间最多允许隔多少个无信号交易日仍算同一簇，"
            f"默认：{base.DEFAULT_CLUSTER_MAX_GAP}；设为 -1 表示每个信号单独计票。"
        ),
    )
    parser.add_argument(
        "--confidence-window-days",
        type=int,
        default=base.DEFAULT_CONFIDENCE_WINDOW_DAYS,
        help=(
            "按同一品种所有选中因子的信号日期计算置信度时，"
            "回看多少个交易日，默认："
            f"{base.DEFAULT_CONFIDENCE_WINDOW_DAYS}。"
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="只输出胜率表，不生成价格信号图。",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=160,
        help="输出 PNG 图片的 dpi，默认：160。",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "按 low、medium、high 三档置信度分层评估信号胜率。"
            "事件日仍采用信号簇最后一个信号日。"
        )
    )
    add_common_arguments(parser)
    args = parser.parse_args()

    if args.lookahead_days_list:
        args.lookahead_days_list = base.parse_int_list(args.lookahead_days_list)
    elif args.lookahead_days is not None:
        args.lookahead_days_list = [args.lookahead_days]
    else:
        args.lookahead_days_list = base.parse_int_list(
            base.DEFAULT_LOOKAHEAD_DAYS_LIST
        )
    args.lookahead_days_list = sorted(set(args.lookahead_days_list))

    if any(value <= 0 for value in args.lookahead_days_list):
        raise ValueError("lookahead-days 必须为正整数。")
    if args.volatility_window <= 0:
        raise ValueError("volatility-window 必须为正整数。")
    if args.volatility_min_history_days <= 0:
        raise ValueError("volatility-min-history-days 必须为正整数。")
    if args.confidence_window_days <= 0:
        raise ValueError("confidence-window-days 必须为正整数。")

    args.output_dir = args.output_dir.resolve()
    return args


def run_evaluation(args):
    runs_dir = args.runs_dir.resolve()
    symbols = base.parse_csv_list(args.symbols)
    factor_ids = base.parse_factor_ids(args.factor_ids)
    factor_files = base.discover_factor_files(
        runs_dir=runs_dir,
        symbols=symbols,
        factor_ids=factor_ids,
    )
    args.confidence_by_symbol_date = base.build_confidence_by_symbol_date(
        factor_files=factor_files,
        lookback_days=args.confidence_window_days,
    )

    rows = []
    baseline_rows = []
    plot_paths = []
    errors = []
    for file_info in factor_files:
        try:
            event_rows, baseline_row, factor_plot_paths = base.evaluate_factor_file(
                file_info,
                args,
            )
            rows.extend(event_rows)
            baseline_rows.extend(baseline_row)
            plot_paths.extend(factor_plot_paths)
        except Exception as exc:
            errors.append((file_info["path"], exc))

    if not rows:
        error_text = "\n".join(f"- {path}: {exc}" for path, exc in errors)
        raise RuntimeError(f"没有可评估的信号事件。\n{error_text}")

    events = pd.DataFrame(rows)
    baselines = pd.DataFrame(baseline_rows)
    summary = base.summarize_events(events, baselines)
    return factor_files, events, summary, plot_paths, errors


def keep_low_medium_high(frame):
    return frame[
        frame["confidence_level"]
        .map(base.normalize_confidence_level)
        .isin(CONFIDENCE_LEVELS)
    ].copy()


def save_outputs(events, summary, output_dir):
    tables_dir = output_dir / "tables"
    events_dir = tables_dir / "events"
    summary_dir = tables_dir / "summary"
    events_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    events_path = events_dir / "signal_cluster_events_low_medium_high.csv"
    overall_path = summary_dir / "overall_by_low_medium_high.csv"
    factor_path = summary_dir / "factor_by_low_medium_high.csv"

    output_events = keep_low_medium_high(base.build_output_events(events))
    output_events.to_csv(events_path, index=False)

    summary_output = base.build_output_summary(summary)
    confidence_mask = summary["confidence_level"].isin(CONFIDENCE_LEVELS)
    overall = summary_output[
        summary["row_type"].eq("all_symbols_all_factors_confidence").to_numpy()
        & confidence_mask.to_numpy()
    ].copy()
    factor = summary_output[
        summary["row_type"].eq("all_symbols_factor_confidence").to_numpy()
        & confidence_mask.to_numpy()
    ].copy()
    overall.to_csv(overall_path, index=False)
    factor.to_csv(factor_path, index=False)

    return {
        "events": events_path,
        "overall": overall_path,
        "factor": factor_path,
    }


def print_report(factor_files, summary, output_paths, plot_paths, args, errors):
    aggregate = summary[
        (summary["row_type"] == "all_symbols_all_factors_confidence")
        & summary["confidence_level"].isin(CONFIDENCE_LEVELS)
    ].copy()
    aggregate["_confidence_sort"] = aggregate["confidence_level"].map(
        base.confidence_level_sort_value
    )
    aggregate = aggregate.sort_values(["lookahead_days", "_confidence_sort"])

    print("low/medium/high 分层胜率评估完成。", flush=True)
    print(f"读取因子文件数量：{len(factor_files)}", flush=True)
    for _, row in aggregate.iterrows():
        print(
            "观察窗口："
            f"{int(row['lookahead_days'])} 日；"
            f"置信度：{row['confidence_level']}；"
            f"信号簇事件：{int(row['cluster_events'])}；"
            f"可评价：{int(row['evaluable_events'])}；"
            f"策略胜率：{row['win_rate']:.2%}；"
            f"基准胜率：{row['baseline_win_rate']:.2%}；"
            f"胜率差：{row['win_rate_lift']:.2%}",
            flush=True,
        )
    print(f"事件明细：{output_paths['events']}", flush=True)
    print(f"总览分层汇总：{output_paths['overall']}", flush=True)
    print(f"因子分层汇总：{output_paths['factor']}", flush=True)
    print(f"置信度回看窗口：{args.confidence_window_days} 个交易日", flush=True)
    if not args.skip_plots:
        print(f"信号图数量：{len(plot_paths)}", flush=True)
        print(
            f"信号图目录：{args.output_dir / 'figures' / 'signal_clusters'}",
            flush=True,
        )
    if errors:
        print("\n以下文件读取失败，已跳过：", flush=True)
        for path, exc in errors:
            print(f"- {path}: {exc}", flush=True)


def main():
    args = parse_args()
    factor_files, events, summary, plot_paths, errors = run_evaluation(args)
    output_paths = save_outputs(events, summary, args.output_dir)
    print_report(factor_files, summary, output_paths, plot_paths, args, errors)


if __name__ == "__main__":
    main()
