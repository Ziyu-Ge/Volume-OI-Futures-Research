import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 参数设置
# =========================

symbol = "LC"

history_window = 10 # 看过去多少个交易日作为“历史参照”。
history_rank_threshold = 0.95 # 历史分位数阈值
trend_rank_threshold = 0.6 # 趋势分位数阈值
min_history_days = 10 # 最小历史天数
min_trend_rank_days = 3 # 最小趋势分位数天数

before_end_days = 20 # 趋势结束前观察天数

price_figsize = (12, 6)
rank_figsize = (12, 5)
signal_point_size = 20
plot_dpi = 300

daily_input_path = f"../results/tables/{symbol}_daily_with_trend.csv"
segments_input_path = f"../results/tables/{symbol}_trend_segments.csv"
daily_output_path = f"../results/tables/{symbol}_daily_high_speculation_signal.csv"
summary_output_path = f"../results/tables/{symbol}_high_speculation_summary.csv"
price_figure_path = f"../results/figures/{symbol}_high_speculation_signal_on_price.png"
rank_figure_path = f"../results/figures/{symbol}_speculation_rank_in_trend.png"


# =========================
# 1. 读取已经做好的趋势数据
# =========================

daily = pd.read_csv(daily_input_path)
segments = pd.read_csv(segments_input_path)

daily["date"] = pd.to_datetime(daily["date"])
segments["start_date"] = pd.to_datetime(segments["start_date"])
segments["end_date"] = pd.to_datetime(segments["end_date"])


# =========================
# 2. 计算历史和趋势段内的投机度分位数
# =========================
# 同时满足：
# 1. 今天投机度处于过去 history_window 天的历史高位；
# 2. 今天投机度处于当前趋势段开始以来的段内高位。
# 两个分位数都只使用“今天以前”的数据，避免未来数据泄露。

daily["speculation_rank_history"] = np.nan
daily["speculation_rank_in_trend"] = np.nan
daily["high_speculation_signal"] = 0


for i in range(len(daily)):

    current_speculation = daily.iloc[i]["speculation"]

    start_i = max(0, i - history_window)

    history = daily.iloc[start_i:i]["speculation"]

    if len(history) < min_history_days:
        continue

    daily.loc[daily.index[i], "speculation_rank_history"] = (
        history <= current_speculation
    ).mean()


for _, seg in segments.iterrows():

    segment_id = seg["segment_id"]

    mask = daily["segment_id"] == segment_id

    part = daily[mask].copy()

    if len(part) == 0:
        continue

    # 在当前趋势段内部，计算“从趋势开始到昨天”的投机度分位数
    ranks = []

    for i in range(len(part)):

        current_speculation = part.iloc[i]["speculation"]

        # 从趋势起点到昨天
        history = part.iloc[:i]["speculation"]

        if len(history) < min_trend_rank_days:
            ranks.append(np.nan)
            continue

        # 今天的投机度在当前趋势历史中的分位数
        rank = (history <= current_speculation).mean()

        ranks.append(rank)

    daily.loc[mask, "speculation_rank_in_trend"] = ranks


# 同时处于历史高位和趋势段内高位，才认为是真正的高投机度
daily.loc[
    (
        daily["speculation_rank_history"] >= history_rank_threshold
    ) & (
        daily["speculation_rank_in_trend"] >= trend_rank_threshold
    ),
    "high_speculation_signal"
] = 1


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

    # 趋势结束前 before_end_days 个自然日左右
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

        # 整个趋势段内的高投机度天数
        "high_spec_days_in_trend": part["high_speculation_signal"].sum(),

        # 整个趋势段内，高投机度天数占比
        "high_spec_ratio_in_trend": part["high_speculation_signal"].mean(),

        # 趋势结束前一段时间内，高投机度天数
        "high_spec_days_before_end": before_end["high_speculation_signal"].sum(),

        # 趋势结束前一段时间内，高投机度天数占比
        "high_spec_ratio_before_end": before_end["high_speculation_signal"].mean(),

        # 趋势结束前一段时间内，最高趋势内投机度分位数
        "max_rank_before_end": before_end["speculation_rank_in_trend"].max(),

        # 趋势结束前一段时间内，平均趋势内投机度分位数
        "mean_rank_before_end": before_end["speculation_rank_in_trend"].mean(),

        # 趋势结束前一段时间内，最高历史投机度分位数
        "max_history_rank_before_end": before_end[
            "speculation_rank_history"
        ].max(),

        # 趋势结束前一段时间内，平均历史投机度分位数
        "mean_history_rank_before_end": before_end[
            "speculation_rank_history"
        ].mean()
    })

high_spec_summary = pd.DataFrame(result_rows)


# =========================
# 5. 保存结果表格
# =========================

os.makedirs(os.path.dirname(daily_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

daily.to_csv(daily_output_path, index=False)
high_spec_summary.to_csv(summary_output_path, index=False)

print("预兆一分析完成。")
print(f"每日信号保存为：{daily_output_path}")
print(f"趋势段汇总保存为：{summary_output_path}")

print("\n趋势段汇总预览：")
print(high_spec_summary.head(20))


# =========================
# 6. 画图：价格、投机度高位信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily["close"], label="close")

high_points = daily[daily["high_speculation_signal"] == 1]

plt.scatter(
    high_points["date"],
    high_points["close"],
    s=signal_point_size,
    label="high speculation signal"
)

plt.title(f"{symbol} Close Price and High Speculation Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 7. 画图：趋势段内投机度分位数
# =========================

plt.figure(figsize=rank_figsize)

plt.plot(
    daily["date"],
    daily["speculation_rank_in_trend"],
    label="speculation rank in current trend"
)

plt.plot(
    daily["date"],
    daily["speculation_rank_history"],
    label="speculation rank in history"
)

plt.axhline(
    trend_rank_threshold,
    linestyle="--",
    label="trend threshold"
)
plt.axhline(
    history_rank_threshold,
    linestyle=":",
    label="history threshold"
)

plt.title("Speculation Rank Within Trend and History")
plt.xlabel("Date")
plt.ylabel("Speculation Rank")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(rank_figure_path), exist_ok=True)
plt.savefig(rank_figure_path, dpi=plot_dpi)
plt.close()

print("\n图片保存为：")
print(price_figure_path)
print(rank_figure_path)
