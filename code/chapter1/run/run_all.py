import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


RUN_DIR = Path(__file__).resolve().parent
CODE_DIR = RUN_DIR.parent
PROJECT_ROOT = CODE_DIR.parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "chapter1"
DAILY_DIR = RESULTS_DIR / "tables"


FACTORS = [
    {
        "id": "11",
        "name": "price_up_volume_oi_surge",
        "script": CODE_DIR / "factors" / "11price_up_volume_oi_surge.py",
        "output": RESULTS_DIR / "11_price_up_volume_oi_surge_all_symbols",
    },
    {
        "id": "12",
        "name": "price_up_speculation_up",
        "script": CODE_DIR / "factors" / "12price_up_speculation_up.py",
        "output": RESULTS_DIR / "12_price_up_speculation_up_all_symbols",
    },
    {
        "id": "13",
        "name": "price_up_oi_down",
        "script": CODE_DIR / "factors" / "13price_up_oi_down.py",
        "output": RESULTS_DIR / "13_price_up_oi_down_all_symbols",
    },
    {
        "id": "14",
        "name": "uptrend_crowded_chase",
        "script": CODE_DIR / "factors" / "14uptrend_crowded_chase.py",
        "output": RESULTS_DIR / "14_uptrend_crowded_chase_all_symbols",
    },
]


def main():
    """一键生成日频数据、运行 11-14 号因子，并更新 combined 输出。"""
    run_prepare_data()
    symbols = discover_symbols()
    print(f"品种数量：{len(symbols)}", flush=True)

    for factor in FACTORS:
        run_factor_for_all_symbols(factor, symbols)
        save_all_symbols_summary(factor, symbols)

    run_combined_plot()
    print("\n全部完成。", flush=True)


def run_prepare_data():
    """先准备日频缓存，后续因子都读取这里的 *_daily.csv。"""
    run_python(CODE_DIR / "00_prepare_data.py")


def discover_symbols():
    """从日频缓存目录读取全部品种代码。"""
    if not DAILY_DIR.is_dir():
        raise FileNotFoundError(f"日频目录不存在：{DAILY_DIR}")

    symbols = sorted(
        path.name[:-len("_daily.csv")].upper()
        for path in DAILY_DIR.glob("*_daily.csv")
        if path.is_file()
    )
    if not symbols:
        raise FileNotFoundError(f"日频目录中没有 *_daily.csv：{DAILY_DIR}")

    return symbols


def run_factor_for_all_symbols(factor, symbols):
    """对全部品种运行一个因子脚本。"""
    print(f"\n===== 运行 {factor['id']} {factor['name']} =====", flush=True)
    factor["output"].mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        print(f"[{factor['id']}] {symbol}", flush=True)
        env = os.environ.copy()
        env["SYMBOL"] = symbol
        env["RESULTS_OUTPUT_DIR"] = str(factor["output"])
        env["CHAPTER1_DAILY_DIR"] = str(DAILY_DIR)
        run_python(factor["script"], env=env)


def save_all_symbols_summary(factor, symbols):
    """把单品种 summary 合并成 all_symbols 汇总表。"""
    summary_frames = []
    for symbol in symbols:
        path = (
            factor["output"]
            / "summary"
            / f"{symbol}_{factor['id']}_{factor['name']}_summary.csv"
        )
        summary = pd.read_csv(path)
        summary.insert(0, "symbol", symbol)
        summary_frames.append(summary)

    output = pd.concat(summary_frames, ignore_index=True).sort_values("symbol")
    output_path = (
        factor["output"]
        / "summary"
        / f"all_symbols_{factor['id']}_{factor['name']}_summary.csv"
    )
    output.to_csv(output_path, index=False)
    print(f"汇总表：{output_path}", flush=True)


def run_combined_plot():
    """用 11-14 号因子结果生成 combined 统计表和图。"""
    run_python(CODE_DIR / "plot" / "plot_combined_signals.py")


def run_python(script, env=None):
    """在项目根目录运行一个 Python 脚本。"""
    subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
