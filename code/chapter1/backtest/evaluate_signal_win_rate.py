import argparse
from pathlib import Path

import pandas as pd

import evaluate_signal_win_rate_core as core
import evaluate_signal_win_rate_outputs as outputs


def parse_args():
    """解析命令行参数，并做基础合法性检查。"""
    parser = argparse.ArgumentParser(
        description=(
            "用信号簇最后一天评估信号胜率："
            "未来窗口最后一天相对事件日的回撤达到"
            "前 20 日平均波动率一倍即算胜。"
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=core.DEFAULT_RUNS_DIR,
        help=f"因子结果根目录，默认：{core.DEFAULT_RUNS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=core.DEFAULT_OUTPUT_DIR,
        help=f"评估结果输出目录，默认：{core.DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--factor-ids",
        default=core.DEFAULT_FACTOR_IDS,
        help=f"要评估的因子 ID，逗号分隔，默认：{core.DEFAULT_FACTOR_IDS}",
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
        default=core.DEFAULT_VOLATILITY_WINDOW,
        help="计算前 N 日平均绝对日收益率，默认：20。",
    )
    parser.add_argument(
        "--volatility-min-history-days",
        type=int,
        default=core.DEFAULT_VOLATILITY_MIN_HISTORY_DAYS,
        help="计算平均波动率时最少需要多少个历史日收益，默认：20。",
    )
    parser.add_argument(
        "--cluster-max-gap",
        type=int,
        default=core.DEFAULT_CLUSTER_MAX_GAP,
        help=(
            "两个信号之间最多允许隔多少个无信号交易日仍算同一簇，"
            f"默认：{core.DEFAULT_CLUSTER_MAX_GAP}；设为 -1 表示每个信号单独计票。"
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

    args = parser.parse_args()
    normalize_args(args)
    return args


def normalize_args(args):
    """补齐默认参数，并把字符串列表转成 Python 列表。"""
    if args.lookahead_days_list:
        args.lookahead_days_list = core.parse_int_list(args.lookahead_days_list)
    elif args.lookahead_days is not None:
        args.lookahead_days_list = [args.lookahead_days]
    else:
        args.lookahead_days_list = core.parse_int_list(
            core.DEFAULT_LOOKAHEAD_DAYS_LIST
        )
    args.lookahead_days_list = sorted(set(args.lookahead_days_list))

    if any(value <= 0 for value in args.lookahead_days_list):
        raise ValueError("lookahead-days 必须为正整数。")
    if args.volatility_window <= 0:
        raise ValueError("volatility-window 必须为正整数。")
    if args.volatility_min_history_days <= 0:
        raise ValueError("volatility-min-history-days 必须为正整数。")

    args.output_dir = args.output_dir.resolve()


def run_evaluation(args):
    """批量评估所有因子文件。"""
    factor_files = core.discover_factor_files(
        runs_dir=args.runs_dir.resolve(),
        factor_ids=core.parse_factor_ids(args.factor_ids),
    )

    rows = []
    baseline_rows = []
    plot_paths = []
    errors = []

    for file_info in factor_files:
        try:
            daily, event_rows, baseline_row, rows_by_lookahead = (
                core.evaluate_factor_file(file_info, args)
            )
            rows.extend(event_rows)
            baseline_rows.extend(baseline_row)

            # 图只是检查用，正式胜率只依赖 CSV。
            if not args.skip_plots:
                for lookahead_days, lookahead_rows in rows_by_lookahead.items():
                    plot_path = outputs.plot_signal_clusters(
                        daily=daily,
                        event_rows=lookahead_rows,
                        args=args,
                        lookahead_days=lookahead_days,
                    )
                    if plot_path is not None:
                        plot_paths.append(plot_path)
        except Exception as exc:
            errors.append((file_info["path"], exc))

    if not rows:
        error_text = "\n".join(f"- {path}: {exc}" for path, exc in errors)
        raise RuntimeError(f"没有可评估的信号事件。\n{error_text}")

    events = pd.DataFrame(rows)
    baselines = pd.DataFrame(baseline_rows)
    summary = core.summarize_events(events, baselines)
    return factor_files, events, summary, plot_paths, errors


def main():
    args = parse_args()
    factor_files, events, summary, plot_paths, errors = run_evaluation(args)
    output_paths = outputs.save_outputs(events, summary, args.output_dir)
    outputs.print_report(
        factor_files=factor_files,
        summary=summary,
        output_paths=output_paths,
        plot_paths=plot_paths,
        args=args,
        errors=errors,
    )


if __name__ == "__main__":
    main()
