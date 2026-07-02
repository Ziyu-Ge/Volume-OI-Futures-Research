import os
from datetime import time

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from config import SYMBOL as CONFIG_SYMBOL


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.abspath(
    os.path.expanduser(
        os.environ.get(
            "RESULTS_OUTPUT_DIR",
            os.path.join(project_root, "results"),
        )
    )
)


# =========================
# 参数设置
# =========================

symbol = os.environ.get("SYMBOL", CONFIG_SYMBOL).upper()
data_path = os.path.join(project_root, "data", f"{symbol}.csv")
daily_output_path = os.path.join(
    results_dir,
    "tables",
    "daily",
    f"{symbol}_daily.csv",
)

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


def validate_columns(frame):
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            "分钟数据缺少必要字段："
            + ", ".join(sorted(missing_columns))
        )


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


# =========================
# 1. 读取分钟数据
# =========================

df = pd.read_csv(data_path)
validate_columns(df)

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
invalid_datetime_count = int(df["datetime"].isna().sum())
if invalid_datetime_count:
    raise ValueError(f"datetime 解析失败行数：{invalid_datetime_count}")

df = df.sort_values("datetime").reset_index(drop=True)


# =========================
# 2. 按交易日标记分钟数据
# =========================

trading_calendar = infer_day_session_calendar(df)
df["date"], shifted_row_count, fallback_evening_count = map_to_trading_dates(
    df["datetime"],
    trading_calendar,
)


# =========================
# 3. 分钟数据聚合成日频数据
# =========================

daily = (
    df.groupby("date", sort=True)
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


# =========================
# 4. 计算投机度
# =========================

daily.loc[daily["open_interest"] <= 0, "open_interest"] = np.nan
daily["speculation"] = np.log(daily["volume"] / daily["open_interest"])


# =========================
# 5. 保存结果
# =========================

os.makedirs(os.path.dirname(daily_output_path), exist_ok=True)
daily.to_csv(daily_output_path, index=False)

print("日频数据准备完成。")
print("划分方式：按交易日，而不是自然日。")
print(f"品种：{symbol}")
print(f"交易日历起止：{trading_calendar[0].date()} - {trading_calendar[-1].date()}")
print(f"跨自然日归并行数：{shifted_row_count}")
if fallback_evening_count:
    print(
        "警告：部分尾部夜盘没有可匹配的后续日盘交易日，"
        f"已用下一工作日兜底，行数：{fallback_evening_count}"
    )
print(f"日频结果保存为：{daily_output_path}")
print("\n日频数据预览：")
print(daily.head(20))
print("\n日频数据行数：")
print(len(daily))
