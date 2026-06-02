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

history_window = 10  # 用过去多少个交易日计算投机度均值和标准差
min_history_days = 10  # 计算均值和标准差所需的最少历史天数
spec_z_threshold = 2  # SpecZ 高于该阈值，认为投机度极端偏离
rolling_std_ddof = 0  # 标准差自由度，0 表示总体标准差
## 总体标准差 sigma = sqrt( sum((x_i - mean)^2) / N ) 和 
## 样本标准差 s = sqrt( sum((x_i - mean)^2) / (N - 1) )的区别
std_epsilon = 1e-12  # 避免标准差过小导致除零

before_end_days = 5  # 趋势结束前观察天数

price_figsize = (12, 6)
zscore_figsize = (12, 6)
signal_point_size = 22
plot_dpi = 300

tables_dir = os.path.join(project_root, "results", "tables")
figures_dir = os.path.join(project_root, "results", "figures")

daily_input_path = os.path.join(tables_dir, f"{symbol}_daily_with_trend.csv")
segments_input_path = os.path.join(tables_dir, f"{symbol}_trend_segments.csv")
daily_output_path = os.path.join(
    tables_dir,
    f"{symbol}_daily_speculation_extreme_deviation_signal.csv"
)
summary_output_path = os.path.join(
    tables_dir,
    f"{symbol}_speculation_extreme_deviation_summary.csv"
)
price_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_speculation_extreme_deviation_signal_on_price.png"
)
price_zscore_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_price_and_speculation_extreme_deviation_zscore.png"
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
# 2. 计算投机度极端偏离因子
# =========================
# SpecZ_t = (Spec_t - mu_{t,N}) / sigma_{t,N}
# 其中 mu 和 sigma 使用今天以前过去 N 个交易日的数据，避免未来数据泄露。

history_speculation = daily["speculation"].shift(1)

daily["speculation_mean_history"] = (
    history_speculation
    .rolling(window=history_window, min_periods=min_history_days)
    .mean()
)
daily["speculation_std_history"] = (
    history_speculation
    .rolling(window=history_window, min_periods=min_history_days)
    .std(ddof=rolling_std_ddof)
)

valid_std = daily["speculation_std_history"].abs() > std_epsilon

daily["speculation_extreme_deviation_z"] = np.nan
daily.loc[valid_std, "speculation_extreme_deviation_z"] = (
    (
        daily.loc[valid_std, "speculation"]
        - daily.loc[valid_std, "speculation_mean_history"]
    )
    / daily.loc[valid_std, "speculation_std_history"]
)

daily["speculation_extreme_deviation_signal"] = 0
daily.loc[
    daily["speculation_extreme_deviation_z"] > spec_z_threshold,
    "speculation_extreme_deviation_signal"
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
        part["speculation_extreme_deviation_signal"] == 1
    ]
    signal_before_end = before_end[
        before_end["speculation_extreme_deviation_signal"] == 1
    ]

    result_rows.append({
        "segment_id": segment_id,
        "trend": seg["trend"],
        "start_date": seg["start_date"],
        "end_date": seg["end_date"],
        "days": seg["days"],
        "return": seg["return"],

        "mean_spec_z_in_trend": part[
            "speculation_extreme_deviation_z"
        ].mean(),
        "max_spec_z_in_trend": part[
            "speculation_extreme_deviation_z"
        ].max(),
        "mean_spec_std_in_trend": part[
            "speculation_std_history"
        ].mean(),

        "extreme_deviation_days_in_trend": part[
            "speculation_extreme_deviation_signal"
        ].sum(),
        "extreme_deviation_ratio_in_trend": part[
            "speculation_extreme_deviation_signal"
        ].mean(),
        "first_extreme_deviation_date": (
            signal_part["date"].min() if len(signal_part) > 0 else pd.NaT
        ),

        "mean_spec_z_before_end": before_end[
            "speculation_extreme_deviation_z"
        ].mean(),
        "max_spec_z_before_end": before_end[
            "speculation_extreme_deviation_z"
        ].max(),
        "mean_spec_std_before_end": before_end[
            "speculation_std_history"
        ].mean(),

        "extreme_deviation_days_before_end": before_end[
            "speculation_extreme_deviation_signal"
        ].sum(),
        "extreme_deviation_ratio_before_end": before_end[
            "speculation_extreme_deviation_signal"
        ].mean(),
        "first_extreme_deviation_date_before_end": (
            signal_before_end["date"].min()
            if len(signal_before_end) > 0
            else pd.NaT
        )
    })

extreme_deviation_summary = pd.DataFrame(result_rows)


# =========================
# 4. 保存结果表格
# =========================

os.makedirs(os.path.dirname(daily_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

daily.to_csv(daily_output_path, index=False)
extreme_deviation_summary.to_csv(summary_output_path, index=False)

print("投机度极端偏离因子分析完成。")
print(f"每日信号保存为：{daily_output_path}")
print(f"趋势段汇总保存为：{summary_output_path}")

print("\n趋势段汇总预览：")
print(extreme_deviation_summary.head(20))


# =========================
# 5. 画图：价格和投机度极端偏离信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily["close"], label="close", color="tab:blue")

signal_points = daily[daily["speculation_extreme_deviation_signal"] == 1]

plt.scatter(
    signal_points["date"],
    signal_points["close"],
    s=signal_point_size,
    label=f"SpecZ > {spec_z_threshold}",
    color="tab:red"
)

plt.title(f"{symbol} Close Price and Speculation Extreme Deviation Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 6. 画图：价格和投机度标准差倍数
# =========================

fig, ax1 = plt.subplots(figsize=zscore_figsize)

ax1.plot(daily["date"], daily["close"], label="close", color="tab:blue")
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(
    daily["date"],
    daily["speculation_extreme_deviation_z"],
    label="SpecZ",
    color="tab:orange"
)
ax2.axhline(0, color="gray", linewidth=1)
ax2.axhline(
    spec_z_threshold,
    color="tab:red",
    linestyle="--",
    linewidth=1,
    label=f"threshold {spec_z_threshold}"
)
ax2.set_ylabel("Speculation Z-Score", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

plt.title(f"{symbol} Close Price and Speculation Z-Score")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
fig.tight_layout()

plt.savefig(price_zscore_figure_path, dpi=plot_dpi)
plt.close()

print("\n图片保存为：")
print(price_figure_path)
print(price_zscore_figure_path)
