import os

import numpy as np
import pandas as pd

from config import SYMBOL


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

symbol = SYMBOL
day_session_start_minute = 8 * 60
night_session_start_minute = 21 * 60
data_path = os.path.join(project_root, "data", f"{symbol}.csv")
daily_output_path = os.path.join(
    results_dir,
    "tables",
    "daily",
    f"{symbol}_daily.csv",
)

## LC
# excluded_dates = pd.to_datetime(["2023-12-08", "2025-08-11", "2026-01-12"])


def infer_trading_dates(datetimes):
    datetimes = pd.to_datetime(datetimes)
    minute_of_day = datetimes.dt.hour * 60 + datetimes.dt.minute
    calendar_dates = datetimes.dt.normalize()

    # 用文件里实际出现的白盘日期作为交易日历。
    # 这样周五或节假日前夜盘会归到下一条实际交易日。
    day_session_mask = (
        (minute_of_day >= day_session_start_minute) &
        (minute_of_day < night_session_start_minute)
    )
    trading_calendar = np.sort(
        calendar_dates.loc[day_session_mask].unique().astype("datetime64[ns]")
    )
    if len(trading_calendar) == 0:
        trading_calendar = np.sort(
            calendar_dates.unique().astype("datetime64[ns]")
        )

    calendar_values = calendar_dates.to_numpy(dtype="datetime64[ns]")
    trading_values = calendar_values.copy()
    known_trading_day = calendar_dates.isin(trading_calendar).to_numpy()
    roll_forward_mask = (
        (minute_of_day.to_numpy() >= night_session_start_minute) |
        ~known_trading_day
    )

    if roll_forward_mask.any():
        roll_indices = np.flatnonzero(roll_forward_mask)
        insert_positions = np.searchsorted(
            trading_calendar,
            calendar_values[roll_indices],
            side="right",
        )
        mapped_mask = insert_positions < len(trading_calendar)

        trading_values[roll_indices[mapped_mask]] = (
            trading_calendar[insert_positions[mapped_mask]]
        )

        if (~mapped_mask).any():
            fallback_dates = (
                pd.to_datetime(calendar_values[roll_indices[~mapped_mask]]) +
                pd.offsets.BDay(1)
            )
            trading_values[roll_indices[~mapped_mask]] = (
                fallback_dates.to_numpy(dtype="datetime64[ns]")
            )

    return pd.Series(pd.to_datetime(trading_values), index=datetimes.index)


# =========================
# 1. 读取分钟数据
# =========================

df = pd.read_csv(data_path)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
df["date"] = infer_trading_dates(df["datetime"])


# =========================
# 2. 分钟数据聚合成日频数据
# =========================

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
# daily = daily[~daily["date"].isin(excluded_dates)].reset_index(drop=True)


# =========================
# 3. 计算投机度
# =========================

daily.loc[daily["open_interest"] <= 0, "open_interest"] = np.nan
daily["speculation"] = np.log(daily["volume"] / daily["open_interest"])


# =========================
# 4. 保存结果
# =========================

os.makedirs(os.path.dirname(daily_output_path), exist_ok=True)
daily.to_csv(daily_output_path, index=False)

print("日频数据准备完成。")
print(f"日频结果保存为：{daily_output_path}")
print("\n日频数据预览：")
print(daily.head(20))
print("\n日频数据行数：")
print(len(daily))
