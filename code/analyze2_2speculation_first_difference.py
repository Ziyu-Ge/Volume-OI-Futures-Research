import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(project_root, ".matplotlib"))

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# 参数设置
# =========================

symbol = "LC"

diff_days = 1  # 和多少个交易日前相比，计算投机度一阶差分
history_window = 20  # 用过去多少个交易日判断一阶差分是否异常
high_diff_rank_threshold = 0.98  # 一阶差分处于历史高位的分位数阈值
low_diff_rank_threshold = 0.02  # 一阶差分处于历史低位的分位数阈值
large_abs_diff_rank_threshold = 0.95  # 绝对一阶差分处于历史高位的分位数阈值
min_history_days = 10  # 计算历史分位数所需的最少历史天数

before_end_days = 5  # 趋势结束前观察天数

price_figsize = (12, 6)
diff_figsize = (12, 6)
signal_point_size = 22
plot_dpi = 300
clip_plot_quantile_low = 0.01  # 画图时裁剪极端值，避免右轴被少数异常点拉扁
clip_plot_quantile_high = 0.99

tables_dir = os.path.join(project_root, "results", "tables")
figures_dir = os.path.join(project_root, "results", "figures")

daily_input_path = os.path.join(tables_dir, f"{symbol}_daily_with_trend.csv")
segments_input_path = os.path.join(tables_dir, f"{symbol}_trend_segments.csv")
daily_output_path = os.path.join(
    tables_dir,
    f"{symbol}_daily_speculation_first_difference_signal.csv"
)
summary_output_path = os.path.join(
    tables_dir,
    f"{symbol}_speculation_first_difference_summary.csv"
)
price_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_speculation_first_difference_signal_on_price.png"
)
price_diff_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_price_and_speculation_first_difference.png"
)


# =========================
# 1. 读取已经做好的趋势数据
# =========================

daily = pd.read_csv(daily_input_path)
segments = pd.read_csv(segments_input_path)

daily["date"] = pd.to_datetime(daily["date"])
segments["start_date"] = pd.to_datetime(segments["start_date"])
segments["end_date"] = pd.to_datetime(segments["end_date"])

daily = daily.sort_values("date").reset_index(drop=True)


# =========================
# 2. 计算投机度一阶差分
# =========================
# DeltaSpec_t = Spec_t - Spec_{t-1}
# DeltaSpec_t > 0 表示投机度上升，DeltaSpec_t < 0 表示投机度回落。
# AbsDeltaSpec_t = |DeltaSpec_t| 表示投机行为变化幅度。

daily["speculation_first_diff"] = (
    daily["speculation"] - daily["speculation"].shift(diff_days)
)
daily["speculation_abs_first_diff"] = (
    daily["speculation_first_diff"].abs()
)


# =========================
# 3. 判断一阶差分是否处于历史异常位置
# =========================
# 分位数只使用“今天以前”的数据，避免未来数据泄露。

daily["speculation_first_diff_rank"] = np.nan
daily["speculation_abs_first_diff_rank"] = np.nan
daily["speculation_first_diff_rise_signal"] = 0
daily["speculation_first_diff_drop_signal"] = 0
daily["speculation_first_diff_large_abs_signal"] = 0

for i in range(len(daily)):

    current_diff = daily.iloc[i]["speculation_first_diff"]
    current_abs_diff = daily.iloc[i]["speculation_abs_first_diff"]

    if pd.isna(current_diff):
        continue

    start_i = max(0, i - history_window)

    diff_history = daily.iloc[start_i:i]["speculation_first_diff"].dropna()
    abs_diff_history = daily.iloc[
        start_i:i
    ]["speculation_abs_first_diff"].dropna()

    if len(diff_history) < min_history_days:
        continue

    diff_rank = (diff_history <= current_diff).mean()
    abs_diff_rank = (abs_diff_history <= current_abs_diff).mean()

    daily.loc[daily.index[i], "speculation_first_diff_rank"] = diff_rank
    daily.loc[daily.index[i], "speculation_abs_first_diff_rank"] = (
        abs_diff_rank
    )

daily.loc[
    daily["speculation_first_diff_rank"] >= high_diff_rank_threshold,
    "speculation_first_diff_rise_signal"
] = 1

daily.loc[
    daily["speculation_first_diff_rank"] <= low_diff_rank_threshold,
    "speculation_first_diff_drop_signal"
] = 1

daily.loc[
    daily["speculation_abs_first_diff_rank"] >= large_abs_diff_rank_threshold,
    "speculation_first_diff_large_abs_signal"
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

        "mean_first_diff_in_trend": part[
            "speculation_first_diff"
        ].mean(),
        "max_first_diff_in_trend": part[
            "speculation_first_diff"
        ].max(),
        "min_first_diff_in_trend": part[
            "speculation_first_diff"
        ].min(),
        "mean_abs_first_diff_in_trend": part[
            "speculation_abs_first_diff"
        ].mean(),
        "max_abs_first_diff_in_trend": part[
            "speculation_abs_first_diff"
        ].max(),

        "first_diff_rise_days_in_trend": part[
            "speculation_first_diff_rise_signal"
        ].sum(),
        "first_diff_rise_ratio_in_trend": part[
            "speculation_first_diff_rise_signal"
        ].mean(),
        "first_diff_drop_days_in_trend": part[
            "speculation_first_diff_drop_signal"
        ].sum(),
        "first_diff_drop_ratio_in_trend": part[
            "speculation_first_diff_drop_signal"
        ].mean(),
        "large_abs_first_diff_days_in_trend": part[
            "speculation_first_diff_large_abs_signal"
        ].sum(),
        "large_abs_first_diff_ratio_in_trend": part[
            "speculation_first_diff_large_abs_signal"
        ].mean(),

        "mean_first_diff_before_end": before_end[
            "speculation_first_diff"
        ].mean(),
        "max_first_diff_before_end": before_end[
            "speculation_first_diff"
        ].max(),
        "min_first_diff_before_end": before_end[
            "speculation_first_diff"
        ].min(),
        "mean_abs_first_diff_before_end": before_end[
            "speculation_abs_first_diff"
        ].mean(),
        "max_abs_first_diff_before_end": before_end[
            "speculation_abs_first_diff"
        ].max(),

        "first_diff_rise_days_before_end": before_end[
            "speculation_first_diff_rise_signal"
        ].sum(),
        "first_diff_rise_ratio_before_end": before_end[
            "speculation_first_diff_rise_signal"
        ].mean(),
        "first_diff_drop_days_before_end": before_end[
            "speculation_first_diff_drop_signal"
        ].sum(),
        "first_diff_drop_ratio_before_end": before_end[
            "speculation_first_diff_drop_signal"
        ].mean(),
        "large_abs_first_diff_days_before_end": before_end[
            "speculation_first_diff_large_abs_signal"
        ].sum(),
        "large_abs_first_diff_ratio_before_end": before_end[
            "speculation_first_diff_large_abs_signal"
        ].mean()
    })

first_diff_summary = pd.DataFrame(result_rows)


# =========================
# 5. 保存结果表格
# =========================

os.makedirs(os.path.dirname(daily_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

daily.to_csv(daily_output_path, index=False)
first_diff_summary.to_csv(summary_output_path, index=False)

print("投机度一阶差分因子分析完成。")
print(f"每日信号保存为：{daily_output_path}")
print(f"趋势段汇总保存为：{summary_output_path}")

print("\n趋势段汇总预览：")
print(first_diff_summary.head(20))


# =========================
# 6. 画图：价格和投机度一阶差分信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily["close"], label="close", color="tab:blue")

rise_points = daily[daily["speculation_first_diff_rise_signal"] == 1]
drop_points = daily[daily["speculation_first_diff_drop_signal"] == 1]

plt.scatter(
    rise_points["date"],
    rise_points["close"],
    s=signal_point_size,
    label="large positive DeltaSpec",
    color="tab:red"
)

plt.scatter(
    drop_points["date"],
    drop_points["close"],
    s=signal_point_size,
    label="large negative DeltaSpec",
    color="tab:green"
)

plt.title(f"{symbol} Close Price and Speculation First Difference Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 7. 画图：价格和投机度一阶差分
# =========================

first_diff_for_plot = daily["speculation_first_diff"].copy()
first_diff_low = first_diff_for_plot.quantile(clip_plot_quantile_low)
first_diff_high = first_diff_for_plot.quantile(clip_plot_quantile_high)
first_diff_for_plot = first_diff_for_plot.clip(
    lower=first_diff_low,
    upper=first_diff_high
)

fig, ax1 = plt.subplots(figsize=diff_figsize)

ax1.plot(daily["date"], daily["close"], label="close", color="tab:blue")
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(
    daily["date"],
    first_diff_for_plot,
    label="DeltaSpec clipped",
    color="tab:orange"
)
ax2.axhline(0, color="gray", linewidth=1)
ax2.set_ylabel("Speculation First Difference", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

plt.title(f"{symbol} Close Price and Speculation First Difference")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
fig.tight_layout()

plt.savefig(price_diff_figure_path, dpi=plot_dpi)
plt.close()

print("\n图片保存为：")
print(price_figure_path)
print(price_diff_figure_path)
