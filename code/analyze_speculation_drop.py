import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 读取已经做好的趋势数据
# =========================

daily = pd.read_csv("../results/tables/LC_daily_with_trend.csv")
segments = pd.read_csv("../results/tables/LC_trend_segments.csv")

daily["date"] = pd.to_datetime(daily["date"])
segments["start_date"] = pd.to_datetime(segments["start_date"])
segments["end_date"] = pd.to_datetime(segments["end_date"])


# =========================
# 2. 计算投机度从高位回落
# =========================
# 投机度回落 = 过去20日最高投机度 - 当天投机度
# 如果这个值 > 0.3，说明投机度已经从高位明显下降

window = 10

daily["speculation_max_20"] = daily["speculation"].rolling(window).max()

daily["speculation_drop_20"] = (
    daily["speculation_max_20"] - daily["speculation"]
)

daily["speculation_drop_signal"] = 0
daily.loc[daily["speculation_drop_20"] > 1, "speculation_drop_signal"] = 1


# =========================
# 3. 只保留趋势段内的数据
# =========================

trend_daily = daily[daily["trend"] != "no_trend"].copy()


# =========================
# 4. 统计每一段趋势中，投机度回落信号出现情况
# =========================

result_rows = []

for _, seg in segments.iterrows():

    segment_id = seg["segment_id"]

    part = trend_daily[trend_daily["segment_id"] == segment_id].copy()

    if len(part) == 0:
        continue

    # 趋势结束日前20个自然日左右的数据
    # 这里写 20 days 是为了简单直观
    end_date = seg["end_date"]
    before_end = part[part["date"] >= end_date - pd.Timedelta(days=20)]

    result_rows.append({
        "segment_id": segment_id,
        "trend": seg["trend"],
        "start_date": seg["start_date"],
        "end_date": seg["end_date"],
        "days": seg["days"],
        "return": seg["return"],

        # 整个趋势段内，投机度回落信号出现天数
        "drop_signal_days_in_trend": part["speculation_drop_signal"].sum(),

        # 整个趋势段内，投机度回落信号占比
        "drop_signal_ratio_in_trend": part["speculation_drop_signal"].mean(),

        # 趋势结束前，投机度回落信号出现天数
        "drop_signal_days_before_end": before_end["speculation_drop_signal"].sum(),

        # 趋势结束前，投机度回落信号占比
        "drop_signal_ratio_before_end": before_end["speculation_drop_signal"].mean(),

        # 趋势结束前，最大投机度回落幅度
        "max_drop_before_end": before_end["speculation_drop_20"].max(),

        # 趋势结束前，平均投机度回落幅度
        "mean_drop_before_end": before_end["speculation_drop_20"].mean()
    })

drop_summary = pd.DataFrame(result_rows)


# =========================
# 5. 保存结果表格
# =========================

daily.to_csv("../results/tables/LC_daily_speculation_drop_signal.csv", index=False)
drop_summary.to_csv("../results/tables/LC_speculation_drop_summary.csv", index=False)

print("预兆二分析完成。")
print("每日信号保存为：../results/tables/LC_daily_speculation_drop_signal.csv")
print("趋势段汇总保存为：../results/tables/LC_speculation_drop_summary.csv")

print("\n趋势段汇总预览：")
print(drop_summary.head(20))


# =========================
# 6. 画图：价格与投机度回落信号
# =========================

plt.figure(figsize=(12, 6))

plt.plot(daily["date"], daily["close"], label="close")

signal_points = daily[daily["speculation_drop_signal"] == 1]

plt.scatter(
    signal_points["date"],
    signal_points["close"],
    s=20,
    label="speculation drop signal"
)

plt.title("LC Close Price and Speculation Drop Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

plt.savefig("../results/figures/LC_speculation_drop_signal_on_price.png", dpi=300)
plt.close()


# =========================
# 7. 画图：投机度回落幅度
# =========================

plt.figure(figsize=(12, 5))

plt.plot(daily["date"], daily["speculation_drop_20"], label="20-day speculation drop")
plt.axhline(0.3, linestyle="--", label="0.3 threshold")

plt.title("Speculation Drop from 20-Day High")
plt.xlabel("Date")
plt.ylabel("Speculation Drop")
plt.legend()
plt.tight_layout()

plt.savefig("../results/figures/LC_speculation_drop_20.png", dpi=300)
plt.close()

print("\n图片保存为：")
print("../results/figures/LC_speculation_drop_signal_on_price.png")
print("../results/figures/LC_speculation_drop_20.png")
