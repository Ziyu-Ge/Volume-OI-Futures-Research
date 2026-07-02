import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHAPTER_RESULTS_DIR = PROJECT_ROOT / "results" / "chapter1"
DEFAULT_DAILY_OUTPUT_DIR = CHAPTER_RESULTS_DIR / "tables" / "daily"
REQUIRED_COLUMNS = {
    "datetime",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "total_turnover",
    "open_interest",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 data/ 下全部品种的分钟数据聚合为日频数据。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"分钟数据目录，默认：{DATA_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DAILY_OUTPUT_DIR,
        help=f"日频数据输出目录，默认：{DEFAULT_DAILY_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="单个品种失败后继续处理后续品种，并在最后汇总失败列表。",
    )
    return parser.parse_args()


def discover_data_files(data_dir):
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    data_files = sorted(path for path in data_dir.glob("*.csv") if path.is_file())
    if not data_files:
        raise FileNotFoundError(f"数据目录中没有 CSV 文件：{data_dir}")

    return data_files


def prepare_daily(data_path):
    symbol = data_path.stem.upper()
    df = pd.read_csv(data_path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{data_path} 缺少字段：{','.join(sorted(missing_columns))}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date

    daily = (
        df.groupby("date")
        .agg({
            "open": "first",
            "close": "last",
            "high": "max",
            "low": "min",
            "volume": "sum",
            "total_turnover": "sum",
            "open_interest": "last",
        })
        .reset_index()
    )

    daily["date"] = pd.to_datetime(daily["date"])
    daily.loc[daily["open_interest"] <= 0, "open_interest"] = np.nan
    daily["speculation"] = np.log(daily["volume"] / daily["open_interest"])

    return symbol, daily


def save_daily(symbol, daily, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}_daily.csv"
    daily.to_csv(output_path, index=False)
    return output_path


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    data_files = discover_data_files(data_dir)
    failures = []
    successes = []

    print(f"分钟数据目录：{data_dir}", flush=True)
    print(f"日频输出目录：{output_dir}", flush=True)
    print(f"待处理品种数量：{len(data_files)}", flush=True)

    for data_path in data_files:
        symbol = data_path.stem.upper()
        try:
            daily_symbol, daily = prepare_daily(data_path)
            output_path = save_daily(daily_symbol, daily, output_dir)
            successes.append(daily_symbol)
            print(
                f"[{daily_symbol}] 日频数据准备完成："
                f"{len(daily)} 行 -> {output_path}",
                flush=True,
            )
        except Exception as exc:
            if not args.keep_going:
                raise

            failures.append((symbol, exc))
            print(f"[{symbol}] 处理失败：{exc}", flush=True)

    print("\n全部日频数据准备完成。", flush=True)
    print(f"成功品种数量：{len(successes)}", flush=True)
    print(f"日频输出目录：{output_dir}", flush=True)

    if failures:
        print(f"失败品种数量：{len(failures)}", flush=True)
        for symbol, exc in failures:
            print(f"- {symbol}: {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
