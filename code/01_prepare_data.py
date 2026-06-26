import os

import numpy as np
import pandas as pd

from config import SYMBOL


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =========================
# 参数设置
# =========================

symbol = os.environ.get("SYMBOL", SYMBOL)
bar_minutes = 15
night_session_start_hour = 20

data_path = os.path.join(project_root, "data", f"{symbol}.csv")
bar15_output_path = os.path.join(
    project_root,
    "results",
    "tables",
    "15min",
    f"{symbol}_15min.csv",
)


def build_trading_day_id(df):
    datetime = df["datetime"]
    calendar_date = datetime.dt.normalize()
    minute_of_day = datetime.dt.hour * 60 + datetime.dt.minute

    previous_calendar_date = calendar_date.shift(1)
    previous_minute_of_day = minute_of_day.shift(1)

    first_row = previous_minute_of_day.isna()
    night_start_minute = night_session_start_hour * 60
    has_night_session = (minute_of_day >= night_start_minute).any()

    if has_night_session:
        night_session_start = (
            (minute_of_day >= night_start_minute) &
            (
                first_row |
                (previous_minute_of_day < night_start_minute) |
                (calendar_date != previous_calendar_date)
            )
        )
        day_session_start_without_night = (
            (minute_of_day < night_start_minute) &
            (calendar_date != previous_calendar_date) &
            (previous_minute_of_day < night_start_minute)
        )
        new_trading_day = (
            first_row |
            night_session_start |
            day_session_start_without_night
        )
    else:
        new_trading_day = first_row | (calendar_date != previous_calendar_date)

    return new_trading_day.cumsum() - 1


# =========================
# 1. 读取分钟数据
# =========================

minute = pd.read_csv(data_path)
minute = minute.drop(
    columns=[
        column for column in minute.columns
        if column.startswith("Unnamed:")
    ],
    errors="ignore",
)
minute["datetime"] = pd.to_datetime(minute["datetime"])
minute = minute.sort_values("datetime").reset_index(drop=True)


# =========================
# 2. 按夜盘开始划分交易日
# =========================

minute["trading_day_id"] = build_trading_day_id(minute)
minute["trading_day_start_datetime"] = (
    minute
    .groupby("trading_day_id")["datetime"]
    .transform("first")
)
minute["trading_day"] = minute[
    "trading_day_start_datetime"
].dt.normalize()


# =========================
# 3. 交易日内每 15 根分钟线聚合一次
# =========================

minute["bar_index_in_trading_day"] = (
    minute
    .groupby("trading_day_id")
    .cumcount() // bar_minutes
)

bar15 = (
    minute
    .groupby(
        ["trading_day_id", "bar_index_in_trading_day"],
        as_index=False,
    )
    .agg({
        "trading_day": "first",
        "trading_day_start_datetime": "first",
        "datetime": ["first", "last", "size"],
        "open": "first",
        "close": "last",
        "high": "max",
        "low": "min",
        "volume": "sum",
        "total_turnover": "sum",
        "open_interest": "last",
    })
)

bar15.columns = [
    "trading_day_id",
    "bar_index_in_trading_day",
    "trading_day",
    "trading_day_start_datetime",
    "bar_start_datetime",
    "bar_end_datetime",
    "source_minutes",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "total_turnover",
    "open_interest",
]
bar15["bars_in_trading_day"] = (
    bar15
    .groupby("trading_day_id")["bar_index_in_trading_day"]
    .transform("count")
)

# 兼容现有因子脚本里的 date 字段；这里的 date 表示 15min bar 结束时间。
bar15["date"] = bar15["bar_end_datetime"]


# =========================
# 4. 计算投机度
# =========================

bar15.loc[bar15["open_interest"] <= 0, "open_interest"] = np.nan
safe_volume = bar15["volume"].where(bar15["volume"] > 0, np.nan)
bar15["speculation"] = np.log(safe_volume / bar15["open_interest"])


# =========================
# 5. 保存结果
# =========================

output_columns = [
    "date",
    "trading_day",
    "trading_day_id",
    "bar_index_in_trading_day",
    "bars_in_trading_day",
    "trading_day_start_datetime",
    "bar_start_datetime",
    "bar_end_datetime",
    "source_minutes",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "total_turnover",
    "open_interest",
    "speculation",
]

os.makedirs(os.path.dirname(bar15_output_path), exist_ok=True)
bar15[output_columns].to_csv(bar15_output_path, index=False)

incomplete_bar_count = int((bar15["source_minutes"] != bar_minutes).sum())

print("15min 数据准备完成。")
print(f"15min 结果保存为：{bar15_output_path}")
print(f"原始分钟行数：{len(minute)}")
print(f"15min 行数：{len(bar15)}")
print(f"交易日数量：{bar15['trading_day_id'].nunique()}")
print(f"非 {bar_minutes} 分钟完整 bar 数量：{incomplete_bar_count}")
print("\n每个交易日 15min bar 数量分布：")
print(bar15.groupby("trading_day_id").size().describe())
print("\n15min 数据预览：")
print(bar15[output_columns].head(20))
