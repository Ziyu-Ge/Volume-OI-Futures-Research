import pandas as pd
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
# 2. 计算过去20日价格高低点和投机度高点
# =========================
# 注意：
# 这里用 shift(1)，表示只看今天以前的过去20天。
# 这样才是真正的“今天价格是否创新高”。

window = 30

daily["close_max_30"] = daily["close"].shift(1).rolling(window).max()
daily["close_min_30"] = daily["close"].shift(1).rolling(window).min()
daily["speculation_max_30"] = daily["speculation"].shift(1).rolling(window).max()


# =========================
# 3. 定义价格和投机度背离信号
# =========================
# 上涨趋势：
# 今天收盘价 > 过去20日最高收盘价
# 但是今天投机度 < 过去20日最高投机度
#
# 下跌趋势：
# 今天收盘价 < 过去20日最低收盘价
# 但是今天投机度 < 过去20日最高投机度

daily["divergence_signal"] = 0

# 上涨趋势中的背离：价格创新高，但投机度没有创新高
up_condition = (
    (daily["trend"] == "up_trend") &
    (daily["close"] > daily["close_max_20"]) &
    (daily["speculation"] < daily["speculation_max_20"])
)

# 下跌趋势中的背离：价格创新低，但投机度没有跟着变强
down_condition = (
    (daily["trend"] == "down_trend") &
    (daily["close"] < daily["close_min_20"]) &
    (daily["speculation"] < daily["speculation_max_20"])
)

daily.loc[up_condition, "divergence_signal"] = 1
daily.loc[down_condition, "divergence_signal"] = 1


# =========================
# 4. 只保留趋势段内的数据
# =========================

trend_daily = daily[daily["trend"] != "no_trend"].copy()


# =========================
# 5. 统计每一段趋势中，背离信号出现情况
# =========================

result_rows = []

for _, seg in segments.iterrows():

    segment_id = seg["segment_id"]

    part = trend_daily[trend_daily["segment_id"] == segment_id].copy()

    if len(part) == 0:
        continue

    # 趋势结束前30个自然日左右
    end_date = seg["end_date"]
    before_end = part[part["date"] >= end_date - pd.Timedelta(days=30)]

    result_rows.append({
        "segment_id": segment_id,
        "trend": seg["trend"],
        "start_date": seg["start_date"],
        "end_date": seg["end_date"],
        "days": seg["days"],
        "return": seg["return"],

        # 整个趋势段内，背离信号出现天数
        "divergence_days_in_trend": part["divergence_signal"].sum(),

        # 整个趋势段内，背离信号占比
        "divergence_ratio_in_trend": part["divergence_signal"].mean(),

        # 趋势结束前，背离信号出现天数
        "divergence_days_before_end": before_end["divergence_signal"].sum(),

        # 趋势结束前，背离信号占比
        "divergence_ratio_before_end": before_end["divergence_signal"].mean()
    })

divergence_summary = pd.DataFrame(result_rows)


# =========================
# 6. 保存结果表格
# =========================

daily.to_csv("../results/tables/LC_daily_divergence_signal.csv", index=False)
divergence_summary.to_csv("../results/tables/LC_divergence_summary.csv", index=False)

print("预兆三分析完成。")
print("每日信号保存为：../results/tables/LC_daily_divergence_signal.csv")
print("趋势段汇总保存为：../results/tables/LC_divergence_summary.csv")

print("\n趋势段汇总预览：")
print(divergence_summary.head(20))


# =========================
# 7. 画图：价格和背离信号
# =========================

plt.figure(figsize=(12, 6))

plt.plot(daily["date"], daily["close"], label="close")

signal_points = daily[daily["divergence_signal"] == 1]

plt.scatter(
    signal_points["date"],
    signal_points["close"],
    s=25,
    label="price-speculation divergence"
)

plt.title("LC Close Price and Price-Speculation Divergence Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

plt.savefig("../results/figures/LC_divergence_signal_on_price.png", dpi=300)
plt.close()


# =========================
# 8. 画图：投机度
# =========================

plt.figure(figsize=(12, 5))

plt.plot(daily["date"], daily["speculation"], label="speculation")
plt.plot(daily["date"], daily["speculation_max_20"], label="20-day max speculation")

plt.title("Speculation and 20-Day Max Speculation")
plt.xlabel("Date")
plt.ylabel("Speculation")
plt.legend()
plt.tight_layout()

plt.savefig("../results/figures/LC_speculation_vs_20day_max.png", dpi=300)
plt.close()

print("\n图片保存为：")
print("../results/figures/LC_divergence_signal_on_price.png")
print("../results/figures/LC_speculation_vs_20day_max.png")