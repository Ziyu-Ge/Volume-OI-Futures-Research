import argparse
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from core.indicators import add_speculation
from core.paths import DAILY_DIR, DATA_DIR, HOURLY_DIR


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
EVENING_ROLLOVER_TIME = time(20, 0)
DAY_START = time(8, 0)
DAY_END = time(16, 0)


@dataclass(frozen=True)
class PrepareResult:
    symbol: str
    output_path: Path
    row_count: int
    shifted_rows: int
    fallback_evening_rows: int


def parse_args():
    parser = argparse.ArgumentParser(description="准备 chapter2 日频/小时频率缓存。")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frequency", choices=["daily", "hourly"], default="daily")
    parser.add_argument("--symbol", action="append", help="可重复传入，默认全部品种。")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def discover_files(data_dir, symbols=None):
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")
    wanted = {symbol.strip().upper() for symbol in symbols} if symbols else None
    files = []
    for path in sorted(data_dir.glob("*.csv")):
        if wanted is None or path.stem.upper() in wanted:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"未找到可处理 CSV：{data_dir}")
    return files


def load_minute_data(path):
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段：{','.join(sorted(missing))}")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    bad_rows = int(frame["datetime"].isna().sum())
    if bad_rows:
        raise ValueError(f"{path} datetime 解析失败行数：{bad_rows}")
    return frame.sort_values("datetime").reset_index(drop=True)


def infer_trading_calendar(frame):
    times = frame["datetime"].dt.time
    day_mask = (times >= DAY_START) & (times <= DAY_END)
    dates = pd.DatetimeIndex(
        frame.loc[day_mask, "datetime"].dt.normalize().dropna().unique()
    ).sort_values()
    if dates.empty:
        raise ValueError("无法从日盘分钟数据推断交易日历")
    return dates


def map_to_trading_date(datetimes, trading_dates):
    """夜盘归到下一交易日；找不到下一交易日时用下一个工作日兜底。"""
    calendar_dates = datetimes.dt.normalize()
    calendar_values = calendar_dates.to_numpy(dtype="datetime64[ns]")
    trading_values = trading_dates.to_numpy(dtype="datetime64[ns]")
    evening_mask = (datetimes.dt.time >= EVENING_ROLLOVER_TIME).to_numpy()
    mapped = calendar_values.copy()

    day_mask = ~evening_mask
    if day_mask.any():
        positions = np.searchsorted(trading_values, calendar_values[day_mask], "left")
        valid = positions < len(trading_values)
        day_mapped = calendar_values[day_mask].copy()
        day_mapped[valid] = trading_values[positions[valid]]
        mapped[day_mask] = day_mapped

    fallback_count = 0
    if evening_mask.any():
        evening_dates = calendar_values[evening_mask]
        positions = np.searchsorted(trading_values, evening_dates, "right")
        valid = positions < len(trading_values)
        evening_mapped = evening_dates.copy()
        evening_mapped[valid] = trading_values[positions[valid]]
        fallback_count = int((~valid).sum())
        if fallback_count:
            fallback = pd.to_datetime(evening_dates[~valid]) + BDay(1)
            evening_mapped[~valid] = fallback.to_numpy(dtype="datetime64[ns]")
        mapped[evening_mask] = evening_mapped

    mapped_dates = pd.Series(pd.to_datetime(mapped), index=datetimes.index, name="date")
    shifted_count = int((mapped_dates != calendar_dates).sum())
    return mapped_dates, shifted_count, fallback_count


def aggregate_daily(frame):
    daily = (
        frame.groupby("date", sort=True)
        .agg(
            open=("open", "first"),
            close=("close", "last"),
            high=("high", "max"),
            low=("low", "min"),
            volume=("volume", "sum"),
            total_turnover=("total_turnover", "sum"),
            open_interest=("open_interest", "last"),
        )
        .reset_index()
    )
    daily.loc[daily["open_interest"] <= 0, "open_interest"] = np.nan
    return add_speculation(daily)


def aggregate_hourly(frame):
    data = frame.copy()
    data["bar_time"] = data["datetime"].dt.floor("h")
    hourly = (
        data.groupby("bar_time", sort=True)
        .agg(
            trading_date=("date", "last"),
            open=("open", "first"),
            close=("close", "last"),
            high=("high", "max"),
            low=("low", "min"),
            volume=("volume", "sum"),
            total_turnover=("total_turnover", "sum"),
            open_interest=("open_interest", "last"),
        )
        .reset_index()
        .rename(columns={"bar_time": "date"})
    )
    hourly.loc[hourly["open_interest"] <= 0, "open_interest"] = np.nan
    return add_speculation(hourly)


def prepare_file(path, output_dir, frequency):
    symbol = path.stem.upper()
    minute = load_minute_data(path)
    calendar = infer_trading_calendar(minute)
    minute["date"], shifted, fallback = map_to_trading_date(
        minute["datetime"], calendar
    )
    bars = aggregate_daily(minute) if frequency == "daily" else aggregate_hourly(minute)
    suffix = "daily" if frequency == "daily" else "hourly"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}_{suffix}.csv"
    bars.to_csv(output_path, index=False)
    return PrepareResult(symbol, output_path, len(bars), shifted, fallback)


def main():
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = DAILY_DIR if args.frequency == "daily" else HOURLY_DIR

    failures = []
    for path in discover_files(args.data_dir.resolve(), args.symbol):
        try:
            result = prepare_file(path, output_dir.resolve(), args.frequency)
            print(
                f"[{result.symbol}] {args.frequency} 完成："
                f"{result.row_count} 行 -> {result.output_path}",
                flush=True,
            )
            if result.fallback_evening_rows:
                print(
                    f"[{result.symbol}] 尾部夜盘兜底："
                    f"{result.fallback_evening_rows} 行",
                    flush=True,
                )
        except Exception as exc:
            if not args.keep_going:
                raise
            failures.append((path.stem.upper(), exc))
            print(f"[{path.stem.upper()}] 失败：{exc}", flush=True)

    if failures:
        for symbol, exc in failures:
            print(f"- {symbol}: {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

