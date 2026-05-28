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
# 2. 计算投机度滚动分位数
# =========================
# 看今天的投机度在过去30天里面排在什么位置。
# 如果分位数 > 0.8，说明今天投机度处于过去120天的高位。

window = 21
分位 = 0.9

daily["speculation_rank_120"] = daily["speculation"].rolling(window).rank(pct=True)

daily["high_speculation_signal"] = 0
daily.loc[daily["speculation_rank_120"] > 分位, "high_speculation_signal"] = 1


# =========================
# 3. 只保留趋势段内的数据
# =========================

trend_daily = daily[daily["trend"] != "no_trend"].copy()


# =========================
# 4. 统计每一段趋势中，高投机度信号出现情况
# =========================

result_rows = []

for _, seg in segments.iterrows():

    segment_id = seg["segment_id"]

    part = trend_daily[trend_daily["segment_id"] == segment_id].copy()

    if len(part) == 0:
        continue

    # 趋势结束日前10天
    end_date = seg["end_date"]
    before_end = part[part["date"] >= end_date - pd.Timedelta(days=20)]

    result_rows.append({
        "segment_id": segment_id,
        "trend": seg["trend"],
        "start_date": seg["start_date"],
        "end_date": seg["end_date"],
        "days": seg["days"],
        "return": seg["return"],

        # 整个趋势段内的高投机度天数
        "high_spec_days_in_trend": part["high_speculation_signal"].sum(),

        # 整个趋势段内，高投机度天数占比
        "high_spec_ratio_in_trend": part["high_speculation_signal"].mean(),

        # 趋势结束前一段时间内，高投机度天数
        "high_spec_days_before_end": before_end["high_speculation_signal"].sum(),

        # 趋势结束前一段时间内，高投机度天数占比
        "high_spec_ratio_before_end": before_end["high_speculation_signal"].mean(),

        # 趋势结束前一段时间内，最高投机度分位数
        "max_rank_before_end": before_end["speculation_rank_120"].max()
    })

high_spec_summary = pd.DataFrame(result_rows)


# =========================
# 5. 保存结果表格
# =========================

daily.to_csv("../results/tables/LC_daily_high_speculation_signal.csv", index=False)
high_spec_summary.to_csv("../results/tables/LC_high_speculation_summary.csv", index=False)

print("预兆一分析完成。")
print("每日信号保存为：../results/tables/LC_daily_high_speculation_signal.csv")
print("趋势段汇总保存为：../results/tables/LC_high_speculation_summary.csv")

print("\n趋势段汇总预览：")
print(high_spec_summary.head(20))


# =========================
# 6. 画图：价格、投机度、投机度高位信号
# =========================

plt.figure(figsize=(12, 6))

plt.plot(daily["date"], daily["close"], label="close")

high_points = daily[daily["high_speculation_signal"] == 1]

plt.scatter(
    high_points["date"],
    high_points["close"],
    s=20,
    label="high speculation signal"
)

plt.title("LC Close Price and High Speculation Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

plt.savefig("../results/figures/LC_high_speculation_signal_on_price.png", dpi=300)
plt.close()


# =========================
# 7. 画图：投机度分位数
# =========================

plt.figure(figsize=(12, 5))

plt.plot(daily["date"], daily["speculation_rank_120"], label="120-day speculation rank")
plt.axhline(0.8, linestyle="--", label="80% threshold")

plt.title("120-Day Rolling Rank of Speculation")
plt.xlabel("Date")
plt.ylabel("Speculation Rank")
plt.legend()
plt.tight_layout()

plt.savefig("../results/figures/LC_speculation_rank_120.png", dpi=300)
plt.close()

print("\n图片保存为：")
print("../results/figures/LC_high_speculation_signal_on_price.png")
print("../results/figures/LC_speculation_rank_120.png")