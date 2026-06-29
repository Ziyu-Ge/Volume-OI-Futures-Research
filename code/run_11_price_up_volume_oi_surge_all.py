import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

FACTOR_ID = "11"
FACTOR_NAME = "price_up_volume_oi_surge"
FACTOR_SCRIPT = CODE_DIR / "factors" / "11price_up_volume_oi_surge.py"
PREPARE_SCRIPT = CODE_DIR / "00_prepare_data.py"
DEFAULT_OUTPUT_DIR = (
    RESULTS_DIR / "11_price_up_volume_oi_surge_all_symbols"
)

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import pandas as pd

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


def parse_symbols(raw_value):
    if raw_value is None:
        return None

    symbols = [
        item.strip().upper()
        for item in raw_value.split(",")
        if item.strip()
    ]

    return symbols or None


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


def build_env(symbol, output_dir):
    env = os.environ.copy()
    env["SYMBOL"] = symbol
    env["FACTOR_ID"] = FACTOR_ID
    env["FACTOR_NAME"] = FACTOR_NAME
    env["RESULTS_OUTPUT_DIR"] = str(output_dir)
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
    return env


def run_script(script_path, symbol, env):
    relative_path = script_path.relative_to(PROJECT_ROOT)
    print(f"\n========== [{symbol}] run {relative_path} ==========", flush=True)

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


def read_factor_summary(symbol, output_dir):
    summary_path = (
        output_dir
        / "tables"
        / "summary"
        / f"{symbol}_{FACTOR_ID}_{FACTOR_NAME}_summary.csv"
    )
    if not summary_path.exists():
        return None

    summary = pd.read_csv(summary_path)
    if "symbol" not in summary.columns:
        summary.insert(0, "symbol", symbol)
    return summary


def save_combined_summaries(symbols, output_dir):
    summaries = []
    for symbol in symbols:
        summary = read_factor_summary(symbol, output_dir)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise FileNotFoundError("没有找到可汇总的 11 号因子 summary。")

    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_summary = combined_summary.sort_values("symbol")

    summary_dir = output_dir / "tables" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        summary_dir
        / f"all_symbols_{FACTOR_ID}_{FACTOR_NAME}_summary.csv"
    )
    combined_summary.to_csv(summary_path, index=False)

    return {
        "summary_path": summary_path,
        "summary_count": len(summaries),
    }


def run_symbol(symbol, args, output_dir):
    env = build_env(symbol, output_dir)

    if not args.collect_only:
        if not args.skip_prepare:
            run_script(PREPARE_SCRIPT, symbol, env)
        if not args.skip_factor:
            run_script(FACTOR_SCRIPT, symbol, env)


def main():
    parser = argparse.ArgumentParser(
        description="运行全部品种的 11 号因子，并输出独立汇总目录。"
    )
    parser.add_argument(
        "--symbols",
        "--symbol",
        dest="symbols",
        help="只运行指定品种，多个品种用逗号分隔，例如：PL,CU。默认运行 data/ 下全部品种。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"结果目录，默认：{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="不重新运行脚本，只基于结果目录中现有 11 号因子 summary 生成汇总。",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="跳过日频数据准备。",
    )
    parser.add_argument(
        "--skip-factor",
        action="store_true",
        help="跳过 11 号因子计算。",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="单个品种失败后继续运行后续品种，并在最后汇总失败列表。",
    )

    args = parser.parse_args()
    symbols = discover_symbols(parse_symbols(args.symbols))
    output_dir = args.output_dir.resolve()
    failures = []

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"本次运行品种数量：{len(symbols)}", flush=True)
    print(f"品种列表：{','.join(symbols)}", flush=True)
    print(f"结果目录：{output_dir}", flush=True)

    for symbol in symbols:
        print(f"\n###### 开始处理品种：{symbol} ######", flush=True)
        try:
            run_symbol(symbol, args, output_dir)
        except Exception as exc:
            if not args.keep_going:
                raise

            failures.append((symbol, exc))
            print(f"\n!!!!!! 品种 {symbol} 处理失败：{exc} !!!!!!", flush=True)

    summary_info = save_combined_summaries(symbols, output_dir)

    print("\n全部任务完成。", flush=True)
    print(f"成功处理品种数量：{len(symbols) - len(failures)}", flush=True)
    print(f"汇总品种数量：{summary_info['summary_count']}", flush=True)
    print(f"汇总表：{summary_info['summary_path']}", flush=True)

    if failures:
        print(f"\n失败品种数量：{len(failures)}", flush=True)
        for symbol, exc in failures:
            print(f"- {symbol}: {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
