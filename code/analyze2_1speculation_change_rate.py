import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 参数设置
# =========================

symbol = "LC"

change_days = 1  # 和多少个交易日前相比，计算投机度变化率
history_window = 20  # 用过去多少个交易日判断变化率是否异常
high_change_rank_threshold = 0.98  # 变化率处于历史高位的分位数阈值
low_change_rank_threshold = 0.02  # 变化率处于历史低位的分位数阈值
min_history_days = 10  # 计算历史分位数所需的最少历史天数

before_end_days = 5  # 趋势结束前观察天数

price_figsize = (12, 6)
change_figsize = (12, 5)
signal_point_size = 20
plot_dpi = 300

daily_input_path = f"../results/tables/{symbol}_daily_with_trend.csv"
segments_input_path = f"../results/tables/{symbol}_trend_segments.csv"
daily_output_path = (
    f"../results/tables/{symbol}_daily_speculation_change_rate_signal.csv"
)
summary_output_path = (
    f"../results/tables/{symbol}_speculation_change_rate_summary.csv"
)
price_figure_path = (
    f"../results/figures/{symbol}_speculation_change_rate_signal_on_price.png"
)
change_figure_path = f"../results/figures/{symbol}_speculation_change_rate.png"
price_change_figure_path = (
    f"../results/figures/{symbol}_price_and_speculation_change_rate.png"
)


# =========================
# 1. 读取已经做好的趋势数据
# =========================

daily = pd.read_csv(daily_input_path)
segments = pd.read_csv(segments_input_path)

daily["date"] = pd.to_datetime(daily["date"])
segments["start_date"] = pd.to_datetime(segments["start_date"])
segments["end_date"] = pd.to_datetime(segments["end_date"])


# =========================
# 2. 计算投机度变化率
# =========================
# speculation = log(volume / open_interest)
# 所以 speculation 的差值，可以还原成 volume / open_interest 的变化率。

daily["speculation_change"] = (
    daily["speculation"] - daily["speculation"].shift(change_days)
)

daily["speculation_change_rate"] = (
    np.exp(daily["speculation_change"]) - 1
)


# =========================
# 3. 判断变化率是否处于历史高位或低位
# =========================
# 分位数只使用“今天以前”的数据，避免未来数据泄露。

daily["speculation_change_rate_rank"] = np.nan
daily["speculation_fast_rise_signal"] = 0
daily["speculation_fast_drop_signal"] = 0

for i in range(len(daily)):

    current_change_rate = daily.iloc[i]["speculation_change_rate"]

    if pd.isna(current_change_rate):
        continue

    start_i = max(0, i - history_window)

    history = daily.iloc[start_i:i]["speculation_change_rate"].dropna()

    if len(history) < min_history_days:
        continue

    rank = (history <= current_change_rate).mean()

    daily.loc[daily.index[i], "speculation_change_rate_rank"] = rank


daily.loc[
    daily["speculation_change_rate_rank"] >= high_change_rank_threshold,
    "speculation_fast_rise_signal"
] = 1

daily.loc[
    daily["speculation_change_rate_rank"] <= low_change_rank_threshold,
    "speculation_fast_drop_signal"
] = 1


# =========================
# 4. 按趋势段汇总
# =========================

trend_daily = daily[daily["trend"] != "no_trend"].copy()

result_rows = []

for _, seg in segments.iterrows():

    segment_id = seg["segment_id"]

    part = trend_daily[trend_daily["segment_id"] == segment_id].copy()

    if len(part) == 0:
        continue

    end_date = seg["end_date"]
    before_end = part[
        part["date"] >= end_date - pd.Timedelta(days=before_end_days)
    ]

    result_rows.append({
        "segment_id": segment_id,
        "trend": seg["trend"],
        "start_date": seg["start_date"],
        "end_date": seg["end_date"],
        "days": seg["days"],
        "return": seg["return"],

        "mean_change_rate_in_trend": part[
            "speculation_change_rate"
        ].mean(),
        "max_change_rate_in_trend": part[
            "speculation_change_rate"
        ].max(),
        "min_change_rate_in_trend": part[
            "speculation_change_rate"
        ].min(),

        "fast_rise_days_in_trend": part[
            "speculation_fast_rise_signal"
        ].sum(),
        "fast_rise_ratio_in_trend": part[
            "speculation_fast_rise_signal"
        ].mean(),
        "fast_drop_days_in_trend": part[
            "speculation_fast_drop_signal"
        ].sum(),
        "fast_drop_ratio_in_trend": part[
            "speculation_fast_drop_signal"
        ].mean(),

        "mean_change_rate_before_end": before_end[
            "speculation_change_rate"
        ].mean(),
        "max_change_rate_before_end": before_end[
            "speculation_change_rate"
        ].max(),
        "min_change_rate_before_end": before_end[
            "speculation_change_rate"
        ].min(),

        "fast_rise_days_before_end": before_end[
            "speculation_fast_rise_signal"
        ].sum(),
        "fast_rise_ratio_before_end": before_end[
            "speculation_fast_rise_signal"
        ].mean(),
        "fast_drop_days_before_end": before_end[
            "speculation_fast_drop_signal"
        ].sum(),
        "fast_drop_ratio_before_end": before_end[
            "speculation_fast_drop_signal"
        ].mean()
    })

change_rate_summary = pd.DataFrame(result_rows)


# =========================
# 5. 保存结果表格
# =========================

os.makedirs(os.path.dirname(daily_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

daily.to_csv(daily_output_path, index=False)
change_rate_summary.to_csv(summary_output_path, index=False)

print("投机度变化率分析完成。")
print(f"每日信号保存为：{daily_output_path}")
print(f"趋势段汇总保存为：{summary_output_path}")

print("\n趋势段汇总预览：")
print(change_rate_summary.head(20))


# =========================
# 6. 画图：价格和变化率信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily["close"], label="close")

fast_rise_points = daily[daily["speculation_fast_rise_signal"] == 1]
fast_drop_points = daily[daily["speculation_fast_drop_signal"] == 1]

plt.scatter(
    fast_rise_points["date"],
    fast_rise_points["close"],
    s=signal_point_size,
    label="fast speculation rise"
)

plt.scatter(
    fast_drop_points["date"],
    fast_drop_points["close"],
    s=signal_point_size,
    label="fast speculation drop"
)

plt.title(f"{symbol} Close Price and Speculation Change Rate Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 7. 画图：价格和投机度变化率
# =========================

change_rate_for_plot = daily["speculation_change_rate"].copy()
change_rate_low = change_rate_for_plot.quantile(0.01)
change_rate_high = change_rate_for_plot.quantile(0.99)
change_rate_for_plot = change_rate_for_plot.clip(
    lower=change_rate_low,
    upper=change_rate_high
)

fig, ax1 = plt.subplots(figsize=price_figsize)

ax1.plot(daily["date"], daily["close"], label="close", color="tab:blue")
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(
    daily["date"],
    change_rate_for_plot,
    label="speculation change rate clipped",
    color="tab:orange"
)
ax2.set_ylabel("Speculation Change Rate", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax2.axhline(0, color="gray", linewidth=1)

plt.title(f"{symbol} Close Price and Speculation Change Rate")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
fig.tight_layout()

plt.savefig(price_change_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 8. 画图：投机度变化率
# =========================

plt.figure(figsize=change_figsize)

plt.plot(
    daily["date"],
    daily["speculation_change_rate"],
    label="speculation change rate"
)

plt.axhline(0, color="gray", linewidth=1)

plt.title("Speculation Change Rate")
plt.xlabel("Date")
plt.ylabel("Change Rate")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(change_figure_path), exist_ok=True)
plt.savefig(change_figure_path, dpi=plot_dpi)
plt.close()

print("\n图片保存为：")
print(price_figure_path)
print(change_figure_path)
print(price_change_figure_path)
