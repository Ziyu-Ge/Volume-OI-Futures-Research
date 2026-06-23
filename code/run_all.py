import argparse
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
FACTOR_DIR = CODE_DIR / "factors"
BACKTEST_DIR = CODE_DIR / "backtest"

FACTOR_UTILS_FILENAME = "volume_price_factor_utils.py"
PREPARE_SCRIPT = CODE_DIR / "00_prepare_data.py"
EDA_SCRIPT = CODE_DIR / "eda.py"
BACKTEST_SCRIPTS = [
    ("sum", BACKTEST_DIR / "backtest_sum.py"),
    ("multi", BACKTEST_DIR / "backtest_multi.py"),
]

RUN_SCRIPT_CODE = textwrap.dedent(
    """
    import importlib.util
    import os
    import runpy
    import sys

    project_root, code_dir, script_path, symbol = sys.argv[1:5]

    os.chdir(project_root)
    os.environ["SYMBOL"] = symbol

    for path in [os.path.dirname(script_path), code_dir, project_root]:
        if path and path not in sys.path:
            sys.path.insert(0, path)

    config_path = os.path.join(code_dir, "config.py")
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config.SYMBOL = symbol
    sys.modules["config"] = config

    runpy.run_path(script_path, run_name="__main__")
    """
)


def parse_csv_filter(raw_value):
    if raw_value is None:
        return None

    values = [
        item.strip().upper()
        for item in raw_value.split(",")
        if item.strip()
    ]

    return values or None


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


def discover_symbols(symbol_filter=None):
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"数据目录不存在：{DATA_DIR}")

    available_symbols = sorted(
        path.stem.upper()
        for path in DATA_DIR.glob("*.csv")
        if path.is_file()
    )
    if not available_symbols:
        raise FileNotFoundError(f"数据目录中没有 CSV 文件：{DATA_DIR}")

    if symbol_filter is None:
        return available_symbols

    available_set = set(available_symbols)
    missing_symbols = [
        symbol
        for symbol in symbol_filter
        if symbol not in available_set
    ]
    if missing_symbols:
        raise FileNotFoundError(
            "以下品种在 data/ 中没有找到对应 CSV："
            f"{','.join(missing_symbols)}"
        )

    return symbol_filter


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


def build_env(symbol, factor_ids=None):
    env = os.environ.copy()
    env["SYMBOL"] = symbol

    if factor_ids is not None:
        env["FACTOR_ID"] = ",".join(sorted(factor_ids))
    else:
        env.pop("FACTOR_ID", None)

    return env


def run_script(script_path, symbol, env):
    relative_path = script_path.relative_to(PROJECT_ROOT)
    print(
        f"\n========== [{symbol}] run {relative_path} ==========",
        flush=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-c",
            RUN_SCRIPT_CODE,
            str(PROJECT_ROOT),
            str(CODE_DIR),
            str(script_path),
            symbol,
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def run_prepare(symbol, env):
    run_script(PREPARE_SCRIPT, symbol, env)


def run_eda(symbol, env):
    if not EDA_SCRIPT.exists():
        raise FileNotFoundError(f"EDA 脚本不存在：{EDA_SCRIPT}")

    run_script(EDA_SCRIPT, symbol, env)


def run_factors(symbol, env, factor_scripts):
    for factor_script in factor_scripts:
        run_script(factor_script, symbol, env)


def run_backtests(symbol, env, backtest_choice):
    ran_backtest = False

    for backtest_name, backtest_script in BACKTEST_SCRIPTS:
        if backtest_choice != "all" and backtest_choice != backtest_name:
            continue

        if not backtest_script.exists():
            if backtest_choice == backtest_name:
                raise FileNotFoundError(f"回测脚本不存在：{backtest_script}")
            continue

        run_script(backtest_script, symbol, env)
        ran_backtest = True

    if not ran_backtest:
        raise FileNotFoundError("没有找到可运行的回测脚本。")


def run_symbol(symbol, args, factor_scripts, factor_ids):
    env = build_env(symbol, factor_ids)

    if not args.skip_prepare:
        run_prepare(symbol, env)

    if args.run_eda:
        run_eda(symbol, env)

    if not args.skip_factors:
        run_factors(symbol, env, factor_scripts)

    if not args.skip_backtest:
        run_backtests(symbol, env, args.backtest)


def main():
    parser = argparse.ArgumentParser(
        description="批量运行全部或指定品种的全部因子和回测。"
    )
    parser.add_argument(
        "--symbols",
        "--symbol",
        dest="symbols",
        help="只运行指定品种，多个品种用逗号分隔，例如：PL,CU。默认运行 data/ 下全部品种。",
    )
    parser.add_argument(
        "--factor-id",
        help="只运行指定因子编号，多个编号用逗号分隔，例如：11,12,14。默认运行全部因子。",
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
        "--skip-backtest",
        action="store_true",
        help="跳过回测。",
    )
    parser.add_argument(
        "--run-eda",
        action="store_true",
        help="额外运行 EDA 脚本。默认不运行。",
    )
    parser.add_argument(
        "--backtest",
        choices=["all", "sum", "multi"],
        default="all",
        help="选择运行哪类回测，默认 all；不存在的回测脚本会在 all 模式下自动跳过。",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="单个品种失败后继续运行后续品种，并在最后汇总失败列表。",
    )

    args = parser.parse_args()
    symbol_filter = parse_csv_filter(args.symbols)
    factor_ids = parse_factor_ids(args.factor_id)
    symbols = discover_symbols(symbol_filter)
    factor_scripts = discover_factor_scripts(factor_ids)
    failures = []

    print(f"本次运行品种数量：{len(symbols)}", flush=True)
    print(f"本次运行因子数量：{len(factor_scripts)}", flush=True)
    print(f"品种列表：{','.join(symbols)}", flush=True)
    print(
        "因子列表："
        + ",".join(script.stem for script in factor_scripts),
        flush=True,
    )

    for symbol in symbols:
        print(f"\n###### 开始运行品种：{symbol} ######", flush=True)
        try:
            run_symbol(symbol, args, factor_scripts, factor_ids)
        except Exception as exc:
            if not args.keep_going:
                raise

            failures.append((symbol, exc))
            print(f"\n!!!!!! 品种 {symbol} 运行失败：{exc} !!!!!!", flush=True)

    print("\n全部任务完成。", flush=True)
    print(f"完成品种数量：{len(symbols) - len(failures)}", flush=True)

    if failures:
        print(f"失败品种数量：{len(failures)}", flush=True)
        for symbol, exc in failures:
            print(f"- {symbol}: {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
