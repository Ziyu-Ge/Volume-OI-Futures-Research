import argparse
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
CODE_DIR = RUN_DIR.parent
PROJECT_ROOT = CODE_DIR.parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "chapter1"
DAILY_DIR = RESULTS_DIR / "tables" / "daily"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import pandas as pd


RUN_SCRIPT_CODE = textwrap.dedent(
    """
    import os
    import runpy
    import sys

    project_root, code_dir, script_path, symbol = sys.argv[1:5]

    os.chdir(project_root)
    os.environ["SYMBOL"] = symbol

    for path in [os.path.dirname(script_path), code_dir, project_root]:
        if path and path not in sys.path:
            sys.path.insert(0, path)

    runpy.run_path(script_path, run_name="__main__")
    """
)


@dataclass(frozen=True)
class FactorRunConfig:
    factor_id: str
    factor_name: str
    factor_script_name: str
    output_dir_name: str

    @property
    def factor_script(self):
        return CODE_DIR / "factors" / self.factor_script_name

    @property
    def default_output_dir(self):
        return RESULTS_DIR / self.output_dir_name


def discover_symbols(daily_dir):
    if not daily_dir.is_dir():
        raise FileNotFoundError(
            f"日频数据目录不存在：{daily_dir}。"
            "请先运行 code/chapter1/00_prepare_data.py。"
        )

    symbols = sorted(
        path.name[:-len("_daily.csv")].upper()
        for path in daily_dir.glob("*_daily.csv")
        if path.is_file()
    )
    if not symbols:
        raise FileNotFoundError(
            f"日频数据目录中没有 *_daily.csv：{daily_dir}。"
            "请先运行 code/chapter1/00_prepare_data.py。"
        )

    return symbols


def discover_symbols_from_summaries(output_dir, config):
    summary_dir = output_dir / "tables" / "summary"
    if not summary_dir.is_dir():
        raise FileNotFoundError(f"summary 目录不存在：{summary_dir}")

    suffix = f"_{config.factor_id}_{config.factor_name}_summary.csv"
    symbols = sorted(
        path.name[:-len(suffix)].upper()
        for path in summary_dir.glob(f"*{suffix}")
        if path.is_file() and not path.name.startswith("all_symbols_")
    )
    if not symbols:
        raise FileNotFoundError(
            f"summary 目录中没有 {config.factor_id} 号因子的单品种汇总："
            f"{summary_dir}"
        )

    return symbols


def build_env(symbol, output_dir, daily_dir, config):
    env = os.environ.copy()
    env["SYMBOL"] = symbol
    env["FACTOR_ID"] = config.factor_id
    env["FACTOR_NAME"] = config.factor_name
    env["RESULTS_OUTPUT_DIR"] = str(output_dir)
    env["CHAPTER1_DAILY_DIR"] = str(daily_dir)
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


def read_factor_summary(symbol, output_dir, config):
    summary_path = (
        output_dir
        / "tables"
        / "summary"
        / f"{symbol}_{config.factor_id}_{config.factor_name}_summary.csv"
    )
    if not summary_path.exists():
        return None

    summary = pd.read_csv(summary_path)
    if "symbol" not in summary.columns:
        summary.insert(0, "symbol", symbol)
    return summary


def save_combined_summaries(symbols, output_dir, config):
    summaries = []
    for symbol in symbols:
        summary = read_factor_summary(symbol, output_dir, config)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise FileNotFoundError(
            f"没有找到可汇总的 {config.factor_id} 号因子 summary。"
        )

    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_summary = combined_summary.sort_values("symbol")

    summary_dir = output_dir / "tables" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        summary_dir
        / f"all_symbols_{config.factor_id}_{config.factor_name}_summary.csv"
    )
    combined_summary.to_csv(summary_path, index=False)

    return {
        "summary_path": summary_path,
        "summary_count": len(summaries),
    }


def run_symbol(symbol, args, output_dir, daily_dir, config):
    env = build_env(symbol, output_dir, daily_dir, config)

    if not args.collect_only and not args.skip_factor:
        run_script(config.factor_script, symbol, env)


def build_parser(config):
    parser = argparse.ArgumentParser(
        description=(
            f"运行全部品种的 {config.factor_id} 号因子，并输出独立汇总目录。"
        )
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
        default=config.default_output_dir,
        help=f"结果目录，默认：{config.default_output_dir}",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            f"不重新运行脚本，只基于结果目录中现有 {config.factor_id} "
            "号因子 summary 生成汇总。"
        ),
    )
    parser.add_argument(
        "--skip-factor",
        action="store_true",
        help=f"跳过 {config.factor_id} 号因子计算，只尝试汇总已有结果。",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="单个品种失败后继续运行后续品种，并在最后汇总失败列表。",
    )
    return parser


def run_factor_all(config):
    parser = build_parser(config)
    args = parser.parse_args()
    daily_dir = args.daily_dir.resolve()
    output_dir = args.output_dir.resolve()
    try:
        symbols = discover_symbols(daily_dir)
    except FileNotFoundError:
        if not args.collect_only and not args.skip_factor:
            raise
        symbols = discover_symbols_from_summaries(output_dir, config)
    failures = []

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"日频缓存目录：{daily_dir}", flush=True)
    print(f"本次运行品种数量：{len(symbols)}", flush=True)
    print(f"品种列表：{','.join(symbols)}", flush=True)
    print(f"结果目录：{output_dir}", flush=True)

    for symbol in symbols:
        print(f"\n###### 开始处理品种：{symbol} ######", flush=True)
        try:
            run_symbol(symbol, args, output_dir, daily_dir, config)
        except Exception as exc:
            if not args.keep_going:
                raise

            failures.append((symbol, exc))
            print(f"\n!!!!!! 品种 {symbol} 处理失败：{exc} !!!!!!", flush=True)

    summary_info = save_combined_summaries(symbols, output_dir, config)

    print("\n全部任务完成。", flush=True)
    print(f"成功处理品种数量：{len(symbols) - len(failures)}", flush=True)
    print(f"汇总品种数量：{summary_info['summary_count']}", flush=True)
    print(f"汇总表：{summary_info['summary_path']}", flush=True)

    if failures:
        print(f"\n失败品种数量：{len(failures)}", flush=True)
        for symbol, exc in failures:
            print(f"- {symbol}: {exc}", flush=True)
        raise SystemExit(1)
