import argparse
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


CHAPTER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CHAPTER_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHAPTER_RESULTS_DIR = PROJECT_ROOT / "results" / "chapter2"
DEFAULT_DAILY_OUTPUT_DIR = CHAPTER_RESULTS_DIR / "tables" / "daily"
DEFAULT_HOURLY_OUTPUT_DIR = CHAPTER_RESULTS_DIR / "tables" / "hourly"

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

# 夜盘可能含有 20:55 的集合竞价，因此用 20:00 作为向下一交易日滚动的边界。
EVENING_SESSION_ROLLOVER_TIME = time(20, 0)
DAY_SESSION_START_TIME = time(8, 0)
DAY_SESSION_END_TIME = time(16, 0)


@dataclass(frozen=True)
class PrepareResult:
    symbol: str
    output_path: Path
    row_count: int
    calendar_start: pd.Timestamp
    calendar_end: pd.Timestamp
    shifted_row_count: int
    fallback_evening_count: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 data/ 下全部品种的分钟数据聚合为 chapter2 bar 数据。"
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
        default=None,
        help=(
            "bar 数据输出目录。默认：daily -> "
            f"{DEFAULT_DAILY_OUTPUT_DIR}；hourly -> {DEFAULT_HOURLY_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--frequency",
        choices=["daily", "hourly"],
        default="daily",
        help="输出频率，默认：daily。",
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


def validate_columns(frame, data_path):
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{data_path} 缺少字段：{', '.join(sorted(missing_columns))}"
        )


def load_minute_data(data_path):
    frame = pd.read_csv(data_path)
    validate_columns(frame, data_path)

    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    invalid_datetime_count = int(frame["datetime"].isna().sum())
    if invalid_datetime_count:
        raise ValueError(f"{data_path} datetime 解析失败行数：{invalid_datetime_count}")

    return frame.sort_values("datetime").reset_index(drop=True)


def infer_day_session_calendar(frame):
    intraday_times = frame["datetime"].dt.time
    day_session_mask = (
        (intraday_times >= DAY_SESSION_START_TIME)
        & (intraday_times <= DAY_SESSION_END_TIME)
    )

    trading_dates = pd.DatetimeIndex(
        frame.loc[day_session_mask, "datetime"]
        .dt.normalize()
        .dropna()
        .unique()
    ).sort_values()

    if trading_dates.empty:
        raise ValueError("无法从日盘分钟数据中推断交易日历。")

    return trading_dates


def map_to_trading_dates(datetimes, trading_dates):
    calendar_dates = datetimes.dt.normalize()
    calendar_values = calendar_dates.to_numpy(dtype="datetime64[ns]")
    trading_values = trading_dates.to_numpy(dtype="datetime64[ns]")
    evening_mask = (
        datetimes.dt.time >= EVENING_SESSION_ROLLOVER_TIME
    ).to_numpy()

    mapped_values = calendar_values.copy()

    non_evening_mask = ~evening_mask
    if non_evening_mask.any():
        non_evening_dates = calendar_values[non_evening_mask]
        positions = np.searchsorted(
            trading_values,
            non_evening_dates,
            side="left",
        )
        valid_positions = positions < len(trading_values)

        non_evening_mapped = non_evening_dates.copy()
        non_evening_mapped[valid_positions] = trading_values[
            positions[valid_positions]
        ]
        mapped_values[non_evening_mask] = non_evening_mapped

    unmatched_evening_count = 0
    if evening_mask.any():
        evening_dates = calendar_values[evening_mask]
        positions = np.searchsorted(
            trading_values,
            evening_dates,
            side="right",
        )
        valid_positions = positions < len(trading_values)

        evening_mapped = evening_dates.copy()
        evening_mapped[valid_positions] = trading_values[
            positions[valid_positions]
        ]

        unmatched_evening_count = int((~valid_positions).sum())
        if unmatched_evening_count:
            fallback_dates = (
                pd.to_datetime(evening_dates[~valid_positions]) + BDay(1)
            )
            evening_mapped[~valid_positions] = fallback_dates.to_numpy(
                dtype="datetime64[ns]"
            )

        mapped_values[evening_mask] = evening_mapped

    mapped_dates = pd.Series(
        pd.to_datetime(mapped_values),
        index=datetimes.index,
        name="date",
    )
    shifted_count = int((mapped_dates != calendar_dates).sum())

    return mapped_dates, shifted_count, unmatched_evening_count


def aggregate_daily(frame):
    daily = (
        frame.groupby("date", sort=True)
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

    daily.loc[daily["open_interest"] <= 0, "open_interest"] = np.nan
    add_speculation_column(daily)

    return daily


def aggregate_hourly(frame):
    hourly_frame = frame.copy()
    hourly_frame["bar_time"] = hourly_frame["datetime"].dt.floor("h")
    hourly = (
        hourly_frame.groupby("bar_time", sort=True)
        .agg({
            "open": "first",
            "close": "last",
            "high": "max",
            "low": "min",
            "volume": "sum",
            "total_turnover": "sum",
            "open_interest": "last",
            "date": "last",
        })
        .reset_index()
        .rename(columns={"bar_time": "date", "date": "trading_date"})
    )

    hourly.loc[hourly["open_interest"] <= 0, "open_interest"] = np.nan
    add_speculation_column(hourly)

    output_columns = [
        "date",
        "trading_date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "total_turnover",
        "open_interest",
        "speculation",
    ]
    return hourly[output_columns]


def add_speculation_column(frame):
    ratio = frame["volume"] / frame["open_interest"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    frame["speculation"] = np.nan
    valid_ratio = ratio > 0
    frame.loc[valid_ratio, "speculation"] = np.log(ratio.loc[valid_ratio])


def aggregate_bars(frame, frequency):
    if frequency == "daily":
        return aggregate_daily(frame)
    if frequency == "hourly":
        return aggregate_hourly(frame)
    raise ValueError(f"不支持的输出频率：{frequency}")


def prepare_bars(data_path, frequency):
    symbol = data_path.stem.upper()
    minute_data = load_minute_data(data_path)
    trading_calendar = infer_day_session_calendar(minute_data)
    (
        minute_data["date"],
        shifted_row_count,
        fallback_evening_count,
    ) = map_to_trading_dates(minute_data["datetime"], trading_calendar)

    bars = aggregate_bars(minute_data, frequency)

    return (
        symbol,
        bars,
        trading_calendar,
        shifted_row_count,
        fallback_evening_count,
    )


def save_bars(symbol, bars, output_dir, frequency):
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "daily" if frequency == "daily" else "hourly"
    output_path = output_dir / f"{symbol}_{suffix}.csv"
    bars.to_csv(output_path, index=False)
    return output_path


def prepare_file(data_path, output_dir, frequency):
    (
        symbol,
        bars,
        trading_calendar,
        shifted_row_count,
        fallback_evening_count,
    ) = prepare_bars(data_path, frequency)
    output_path = save_bars(symbol, bars, output_dir, frequency)

    return PrepareResult(
        symbol=symbol,
        output_path=output_path,
        row_count=len(bars),
        calendar_start=trading_calendar[0],
        calendar_end=trading_calendar[-1],
        shifted_row_count=shifted_row_count,
        fallback_evening_count=fallback_evening_count,
    )


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if args.output_dir is None:
        default_output_dir = (
            DEFAULT_DAILY_OUTPUT_DIR
            if args.frequency == "daily"
            else DEFAULT_HOURLY_OUTPUT_DIR
        )
        output_dir = default_output_dir.resolve()
    else:
        output_dir = args.output_dir.resolve()
    data_files = discover_data_files(data_dir)
    failures = []
    successes = []

    print(f"分钟数据目录：{data_dir}", flush=True)
    print(f"chapter2 输出频率：{args.frequency}", flush=True)
    print(f"chapter2 bar 输出目录：{output_dir}", flush=True)
    print(f"待处理品种数量：{len(data_files)}", flush=True)

    for data_path in data_files:
        symbol = data_path.stem.upper()
        try:
            result = prepare_file(data_path, output_dir, args.frequency)
            successes.append(result)
            print(
                f"[{result.symbol}] bar 数据准备完成："
                f"{result.row_count} 行 -> {result.output_path}",
                flush=True,
            )
            print(
                f"[{result.symbol}] 交易日历："
                f"{result.calendar_start.date()} - "
                f"{result.calendar_end.date()}，"
                f"跨自然日归并行数：{result.shifted_row_count}",
                flush=True,
            )
            if result.fallback_evening_count:
                print(
                    f"[{result.symbol}] 警告：尾部夜盘兜底行数："
                    f"{result.fallback_evening_count}",
                    flush=True,
                )
        except Exception as exc:
            if not args.keep_going:
                raise

            failures.append((symbol, exc))
            print(f"[{symbol}] 处理失败：{exc}", flush=True)

    print("\n全部 bar 数据准备完成。", flush=True)
    print(f"成功品种数量：{len(successes)}", flush=True)
    print(f"bar 输出目录：{output_dir}", flush=True)

    if failures:
        print(f"失败品种数量：{len(failures)}", flush=True)
        for symbol, exc in failures:
            print(f"- {symbol}: {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
