import argparse
import os
import subprocess
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = RUN_DIR.parent
PROJECT_ROOT = CHAPTER_DIR.parents[1]
CHAPTER_RESULTS_DIR = PROJECT_ROOT / "results" / "chapter2"
DAILY_DIR = CHAPTER_RESULTS_DIR / "tables" / "daily"
HOURLY_DIR = CHAPTER_RESULTS_DIR / "tables" / "hourly"
FACTOR_OUTPUT_DIR = (
    CHAPTER_RESULTS_DIR / "24_high_bias_oi_speculation_drop_mixed_all_symbols"
)
BACKTEST_OUTPUT_DIR = (
    CHAPTER_RESULTS_DIR
    / "backtest"
    / "24_high_bias_oi_speculation_drop_mixed_simple"
)
FACTOR_SCRIPT = (
    CHAPTER_DIR / "factors" / "24_high_bias_oi_speculation_drop_mixed.py"
)
BACKTEST_SCRIPT = (
    CHAPTER_DIR
    / "backtest"
    / "backtest_24_high_bias_oi_speculation_drop_mixed_simple.py"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "基于已有 chapter2 日频和小时频率数据运行全部品种的 24 号因子与回测。"
        )
    )
    parser.add_argument(
        "--daily-dir",
        type=Path,
        default=DAILY_DIR,
        help=f"日频缓存目录，默认：{DAILY_DIR}",
    )
    parser.add_argument(
        "--hourly-dir",
        type=Path,
        default=HOURLY_DIR,
        help=f"小时频率缓存目录，默认：{HOURLY_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FACTOR_OUTPUT_DIR,
        help=f"因子结果目录，默认：{FACTOR_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--backtest-output-dir",
        type=Path,
        default=BACKTEST_OUTPUT_DIR,
        help=f"回测结果目录，默认：{BACKTEST_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        help="指定要运行的品种，可重复传入；默认运行全部可用交集品种。",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="因子阶段不重新计算，只基于 output-dir 中已有 summary 生成汇总。",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="单个品种失败后继续运行后续品种，并在最后汇总失败列表。",
    )
    return parser


def build_env(output_dir, daily_dir, hourly_dir):
    env = os.environ.copy()
    env["RESULTS_OUTPUT_DIR"] = str(output_dir)
    env["CHAPTER2_DAILY_DIR"] = str(daily_dir)
    env["CHAPTER2_HOURLY_DIR"] = str(hourly_dir)
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
    (PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)
    (PROJECT_ROOT / ".cache").mkdir(exist_ok=True)
    return env


def run_command(command, env):
    print("\n========== run ==========", flush=True)
    print(" ".join(str(part) for part in command), flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def append_repeated_args(command, option, values):
    if values is None:
        return
    for value in values:
        command.extend([option, value])


def main():
    parser = build_parser()
    args = parser.parse_args()

    daily_dir = args.daily_dir.resolve()
    hourly_dir = args.hourly_dir.resolve()
    output_dir = args.output_dir.resolve()
    backtest_output_dir = args.backtest_output_dir.resolve()
    env = build_env(output_dir, daily_dir, hourly_dir)

    print(f"chapter2 日频目录：{daily_dir}", flush=True)
    print(f"chapter2 小时频率目录：{hourly_dir}", flush=True)
    print(f"chapter2 因子结果目录：{output_dir}", flush=True)
    print(f"chapter2 回测结果目录：{backtest_output_dir}", flush=True)

    factor_command = [
        sys.executable,
        FACTOR_SCRIPT,
        "--daily-dir",
        daily_dir,
        "--hourly-dir",
        hourly_dir,
        "--output-dir",
        output_dir,
    ]
    append_repeated_args(factor_command, "--symbol", args.symbol)
    if args.collect_only:
        factor_command.append("--collect-only")
    if args.keep_going:
        factor_command.append("--keep-going")
    run_command(factor_command, env)

    backtest_command = [
        sys.executable,
        BACKTEST_SCRIPT,
        "--daily-dir",
        daily_dir,
        "--hourly-dir",
        hourly_dir,
        "--output-dir",
        backtest_output_dir,
    ]
    append_repeated_args(backtest_command, "--symbol", args.symbol)
    if args.keep_going:
        backtest_command.append("--keep-going")
    run_command(backtest_command, env)


if __name__ == "__main__":
    main()
