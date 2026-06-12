import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from config import SYMBOL


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
FACTOR_DIR = CODE_DIR / "factors"
BACKTEST_DIR = CODE_DIR / "backtest"

FACTOR_UTILS_FILENAME = "volume_price_factor_utils.py"
BACKTEST_SCRIPTS = [
    ("multi", BACKTEST_DIR / "backtest_multi.py"),
    ("sum", BACKTEST_DIR / "backtest_sum.py"),
]


def parse_factor_ids(raw_value):
    if raw_value is None:
        return None

    factor_ids = {
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    }

    return factor_ids or None


def get_factor_id_from_script(script_path):
    match = re.match(r"^(\d+)", script_path.stem)
    if match is None:
        raise ValueError(f"因子脚本文件名必须以编号开头：{script_path.name}")

    return match.group(1)


def discover_factor_scripts(factor_ids=None):
    scripts = []

    for script_path in sorted(FACTOR_DIR.glob("*.py")):
        if script_path.name == FACTOR_UTILS_FILENAME:
            continue
        if script_path.name.startswith("_"):
            continue

        factor_id = get_factor_id_from_script(script_path)
        if factor_ids is not None and factor_id not in factor_ids:
            continue

        scripts.append(script_path)

    if not scripts:
        raise FileNotFoundError("没有找到需要运行的因子脚本。")

    return scripts


def build_env(factor_ids=None):
    env = os.environ.copy()
    env["SYMBOL"] = SYMBOL

    if factor_ids is not None:
        env["FACTOR_ID"] = ",".join(sorted(factor_ids))

    return env


def run_script(script_path, env):
    relative_path = script_path.relative_to(PROJECT_ROOT)
    print(f"\n========== run {relative_path} ==========", flush=True)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def run_prepare(env):
    run_script(CODE_DIR / "00_prepare_data.py", env)


def run_eda(env):
    run_script(CODE_DIR / "eda.py", env)


def run_factors(env, factor_ids=None):
    factor_scripts = discover_factor_scripts(factor_ids)

    for factor_script in factor_scripts:
        run_script(factor_script, env)

    return factor_scripts


def run_backtests(env, backtest_choice):
    for backtest_name, backtest_script in BACKTEST_SCRIPTS:
        if backtest_choice != "all" and backtest_choice != backtest_name:
            continue

        run_script(backtest_script, env)


def main():
    parser = argparse.ArgumentParser(
        description="依次运行数据准备、EDA、全部因子和回测。"
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="跳过日频数据准备。",
    )
    parser.add_argument(
        "--skip-factors",
        action="store_true",
        help="跳过因子计算。",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="跳过 EDA 图表生成。",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="跳过回测。",
    )
    parser.add_argument(
        "--factor-id",
        help="只运行指定因子编号，多个编号用逗号分隔，例如：11,12,44。",
    )
    parser.add_argument(
        "--backtest",
        choices=["all", "multi", "sum"],
        default="all",
        help="选择运行哪类回测，默认 all。",
    )

    args = parser.parse_args()
    factor_ids = parse_factor_ids(args.factor_id)
    env = build_env(factor_ids)

    print(f"SYMBOL = {SYMBOL}", flush=True)

    if not args.skip_prepare:
        run_prepare(env)

    if not args.skip_eda:
        run_eda(env)

    factor_scripts = []
    if not args.skip_factors:
        factor_scripts = run_factors(env, factor_ids)

    if not args.skip_backtest:
        run_backtests(env, args.backtest)

    print("\n全部任务完成。", flush=True)
    if factor_scripts:
        print(f"本次运行因子数量：{len(factor_scripts)}", flush=True)


if __name__ == "__main__":
    main()
