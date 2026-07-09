import argparse
import os
import subprocess
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = RUN_DIR.parent
PROJECT_ROOT = CHAPTER_DIR.parents[1]
CHAPTER_RESULTS_DIR = PROJECT_ROOT / "results" / "chapter2"
HOURLY_DIR = CHAPTER_RESULTS_DIR / "tables" / "hourly"
FACTOR_OUTPUT_DIR = CHAPTER_RESULTS_DIR / "22_high_bias_oi_drop_hourly_all_symbols"
FACTOR_SCRIPT = CHAPTER_DIR / "factors" / "22_high_bias_oi_drop_hourly.py"


def build_parser():
    parser = argparse.ArgumentParser(
        description="基于已有 chapter2 小时频率数据运行全部品种的 22 号因子。"
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


def build_env(output_dir, hourly_dir):
    env = os.environ.copy()
    env["RESULTS_OUTPUT_DIR"] = str(output_dir)
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


def main():
    parser = build_parser()
    args = parser.parse_args()

    hourly_dir = args.hourly_dir.resolve()
    output_dir = args.output_dir.resolve()
    env = build_env(output_dir, hourly_dir)

    print(f"chapter2 小时频率目录：{hourly_dir}", flush=True)
    print(f"chapter2 因子结果目录：{output_dir}", flush=True)

    factor_command = [
        sys.executable,
        FACTOR_SCRIPT,
        "--hourly-dir",
        hourly_dir,
        "--output-dir",
        output_dir,
    ]
    if args.collect_only:
        factor_command.append("--collect-only")
    if args.keep_going:
        factor_command.append("--keep-going")
    run_command(factor_command, env)


if __name__ == "__main__":
    main()
