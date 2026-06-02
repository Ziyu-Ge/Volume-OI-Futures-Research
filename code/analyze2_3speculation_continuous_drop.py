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
down_window = 5  # 过去多少个交易日内统计投机度下降天数
down_days_threshold = 4  # 过去 down_window 天里，投机度下降天数达到该值则触发信号
min_down_window_days = 3  # 统计下降天数所需的最少历史天数
strictly_negative_diff = True  # True 表示 DeltaSpec < 0 才算下降

before_end_days = 5  # 趋势结束前观察天数

price_figsize = (12, 6)
down_days_figsize = (12, 6)
signal_point_size = 22
plot_dpi = 300

tables_dir = os.path.join(project_root, "results", "tables")
figures_dir = os.path.join(project_root, "results", "figures")

daily_input_path = os.path.join(tables_dir, f"{symbol}_daily_with_trend.csv")
segments_input_path = os.path.join(tables_dir, f"{symbol}_trend_segments.csv")
daily_output_path = os.path.join(
    tables_dir,
    f"{symbol}_daily_speculation_continuous_drop_signal.csv"
)
summary_output_path = os.path.join(
    tables_dir,
    f"{symbol}_speculation_continuous_drop_summary.csv"
)
price_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_speculation_continuous_drop_signal_on_price.png"
)
price_down_days_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_price_and_speculation_continuous_drop.png"
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
# 2. 计算投机度连续回落因子
# =========================
# DeltaSpec_t = Spec_t - Spec_{t-1}
# SpecDownDays_t = sum(I(DeltaSpec_i < 0)), i from t-k+1 to t

daily["speculation_first_diff"] = (
    daily["speculation"] - daily["speculation"].shift(diff_days)
)

if strictly_negative_diff:
    daily["speculation_down_flag"] = (
        daily["speculation_first_diff"] < 0
    ).astype(int)
else:
    daily["speculation_down_flag"] = (
        daily["speculation_first_diff"] <= 0
    ).astype(int)

daily.loc[
    daily["speculation_first_diff"].isna(),
    "speculation_down_flag"
] = np.nan

daily["speculation_down_days"] = (
    daily["speculation_down_flag"]
    .rolling(window=down_window, min_periods=min_down_window_days)
    .sum()
)

down_streak = []
current_streak = 0

for down_flag in daily["speculation_down_flag"]:

    if pd.isna(down_flag):
        current_streak = 0
    elif down_flag == 1:
        current_streak += 1
    else:
        current_streak = 0

    down_streak.append(current_streak)

daily["speculation_down_streak"] = down_streak

daily["speculation_continuous_drop_signal"] = 0
daily.loc[
    daily["speculation_down_days"] >= down_days_threshold,
    "speculation_continuous_drop_signal"
] = 1


# =========================
# 3. 按趋势段汇总
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

    signal_part = part[
        part["speculation_continuous_drop_signal"] == 1
    ]
    signal_before_end = before_end[
        before_end["speculation_continuous_drop_signal"] == 1
    ]

    result_rows.append({
        "segment_id": segment_id,
        "trend": seg["trend"],
        "start_date": seg["start_date"],
        "end_date": seg["end_date"],
        "days": seg["days"],
        "return": seg["return"],

        "mean_down_days_in_trend": part[
            "speculation_down_days"
        ].mean(),
        "max_down_days_in_trend": part[
            "speculation_down_days"
        ].max(),
        "mean_down_streak_in_trend": part[
            "speculation_down_streak"
        ].mean(),
        "max_down_streak_in_trend": part[
            "speculation_down_streak"
        ].max(),
        "continuous_drop_days_in_trend": part[
            "speculation_continuous_drop_signal"
        ].sum(),
        "continuous_drop_ratio_in_trend": part[
            "speculation_continuous_drop_signal"
        ].mean(),
        "first_continuous_drop_date": (
            signal_part["date"].min() if len(signal_part) > 0 else pd.NaT
        ),

        "mean_down_days_before_end": before_end[
            "speculation_down_days"
        ].mean(),
        "max_down_days_before_end": before_end[
            "speculation_down_days"
        ].max(),
        "mean_down_streak_before_end": before_end[
            "speculation_down_streak"
        ].mean(),
        "max_down_streak_before_end": before_end[
            "speculation_down_streak"
        ].max(),
        "continuous_drop_days_before_end": before_end[
            "speculation_continuous_drop_signal"
        ].sum(),
        "continuous_drop_ratio_before_end": before_end[
            "speculation_continuous_drop_signal"
        ].mean(),
        "first_continuous_drop_date_before_end": (
            signal_before_end["date"].min()
            if len(signal_before_end) > 0
            else pd.NaT
        )
    })

continuous_drop_summary = pd.DataFrame(result_rows)


# =========================
# 4. 保存结果表格
# =========================

os.makedirs(os.path.dirname(daily_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

daily.to_csv(daily_output_path, index=False)
continuous_drop_summary.to_csv(summary_output_path, index=False)

print("投机度连续回落因子分析完成。")
print(f"每日信号保存为：{daily_output_path}")
print(f"趋势段汇总保存为：{summary_output_path}")

print("\n趋势段汇总预览：")
print(continuous_drop_summary.head(20))


# =========================
# 5. 画图：价格和投机度连续回落信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily["close"], label="close", color="tab:blue")

signal_points = daily[daily["speculation_continuous_drop_signal"] == 1]

plt.scatter(
    signal_points["date"],
    signal_points["close"],
    s=signal_point_size,
    label=f"SpecDownDays >= {down_days_threshold}",
    color="tab:green"
)

plt.title(f"{symbol} Close Price and Speculation Continuous Drop Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 6. 画图：价格和投机度连续回落天数
# =========================

fig, ax1 = plt.subplots(figsize=down_days_figsize)

ax1.plot(daily["date"], daily["close"], label="close", color="tab:blue")
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.step(
    daily["date"],
    daily["speculation_down_days"],
    where="mid",
    label="SpecDownDays",
    color="tab:orange"
)
ax2.axhline(
    down_days_threshold,
    color="tab:green",
    linestyle="--",
    linewidth=1,
    label=f"threshold {down_days_threshold}"
)
ax2.set_ylabel(
    f"Speculation Down Days in Past {down_window} Days",
    color="tab:orange"
)
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax2.set_ylim(
    -0.2,
    max(down_window, down_days_threshold) + 0.5
)

plt.title(f"{symbol} Close Price and Speculation Continuous Drop")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
fig.tight_layout()

plt.savefig(price_down_days_figure_path, dpi=plot_dpi)
plt.close()

print("\n图片保存为：")
print(price_figure_path)
print(price_down_days_figure_path)
