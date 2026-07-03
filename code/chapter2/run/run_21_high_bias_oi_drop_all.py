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
FACTOR_OUTPUT_DIR = CHAPTER_RESULTS_DIR / "21_high_bias_oi_drop_all_symbols"
BACKTEST_OUTPUT_DIR = (
    CHAPTER_RESULTS_DIR
    / "backtest"
    / "21_high_bias_oi_drop_simple_interest"
)
FACTOR_SCRIPT = CHAPTER_DIR / "factors" / "21_high_bias_oi_drop.py"
BACKTEST_SCRIPT = (
    CHAPTER_DIR / "backtest" / "backtest_21_high_bias_oi_drop_simple.py"
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="基于已有 chapter2 日频数据运行全部品种的 21 号因子。"
    )
    parser.add_argument(
        "--daily-dir",
        type=Path,
        default=DAILY_DIR,
        help=f"日频缓存目录，默认：{DAILY_DIR}",
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
        "--collect-only",
        action="store_true",
        help="不重新计算因子，只基于 output-dir 中已有 summary 生成全品种汇总。",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="单个品种失败后继续运行后续品种，并在最后汇总失败列表。",
    )
    return parser


def build_env(output_dir, daily_dir):
    env = os.environ.copy()
    env["RESULTS_OUTPUT_DIR"] = str(output_dir)
    env["CHAPTER2_DAILY_DIR"] = str(daily_dir)
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


def main():
    parser = build_parser()
    args = parser.parse_args()

    daily_dir = args.daily_dir.resolve()
    output_dir = args.output_dir.resolve()
    backtest_output_dir = args.backtest_output_dir.resolve()
    env = build_env(output_dir, daily_dir)

    print(f"chapter2 日频目录：{daily_dir}", flush=True)
    print(f"chapter2 因子结果目录：{output_dir}", flush=True)
    print(f"chapter2 回测结果目录：{backtest_output_dir}", flush=True)

    factor_command = [
        sys.executable,
        FACTOR_SCRIPT,
        "--daily-dir",
        daily_dir,
        "--output-dir",
        output_dir,
    ]
    if args.collect_only:
        factor_command.append("--collect-only")
    if args.keep_going:
        factor_command.append("--keep-going")
    run_command(factor_command, env)

    backtest_command = [
        sys.executable,
        BACKTEST_SCRIPT,
        "--factor-output-dir",
        output_dir,
        "--output-dir",
        backtest_output_dir,
    ]
    if args.keep_going:
        backtest_command.append("--keep-going")
    run_command(backtest_command, env)


if __name__ == "__main__":
    main()
