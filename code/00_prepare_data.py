import os

import numpy as np
import pandas as pd


# =========================
# 参数设置
# =========================

symbol = "LC"
data_path = f"../data/{symbol}.csv"
daily_output_path = f"../results/tables/daily/{symbol}_daily.csv"

excluded_dates = pd.to_datetime(["2023-12-08", "2025-08-11", "2026-01-12"])


# =========================
# 1. 读取分钟数据
# =========================

df = pd.read_csv(data_path)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
df["date"] = df["datetime"].dt.date


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
daily = daily[~daily["date"].isin(excluded_dates)].reset_index(drop=True)


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
