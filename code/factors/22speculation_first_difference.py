import os

# 当前文件在 code/factors/ 下面
# 所以项目根目录是往上两层
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

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

factor_id = "22"
factor_name = "speculation_first_difference"

diff_days = 1  # 和多少个交易日前相比，计算投机度一阶差分
history_window = 20  # 用过去多少个交易日判断一阶差分是否异常
high_diff_rank_threshold = 0.98  # 一阶差分处于历史高位的分位数阈值
low_diff_rank_threshold = 0.02  # 一阶差分处于历史低位的分位数阈值
large_abs_diff_rank_threshold = 0.95  # 绝对一阶差分处于历史高位的分位数阈值
min_history_days = 10  # 计算历史分位数所需的最少历史天数

# 单因子触发后，建议仓位比例
signal_position_scale = 0.7

price_figsize = (12, 6)
diff_figsize = (12, 6)
signal_point_size = 22
plot_dpi = 300

# 画图时裁剪极端值，避免右轴被少数异常点拉扁
clip_plot_quantile_low = 0.01
clip_plot_quantile_high = 0.99

# =========================
# 输入路径
# =========================

daily_input_path = os.path.join(
    project_root,
    "results",
    "tables",
    "daily",
    f"{symbol}_daily_with_trend.csv"
)

segments_input_path = os.path.join(
    project_root,
    "results",
    "tables",
    "daily",
    f"{symbol}_trend_segments.csv"
)

# 如果还没有把 prepare_data.py 的输出改到 daily 文件夹，则自动使用旧路径
if not os.path.exists(daily_input_path):
    daily_input_path = os.path.join(
        project_root,
        "results",
        "tables",
        f"{symbol}_daily_with_trend.csv"
    )

if not os.path.exists(segments_input_path):
    segments_input_path = os.path.join(
        project_root,
        "results",
        "tables",
        f"{symbol}_trend_segments.csv"
    )

# =========================
# 输出路径
# =========================

factor_output_path = os.path.join(
    project_root,
    "results",
    "tables",
    "factors",
    f"{symbol}_{factor_id}_{factor_name}.csv"
)

signal_output_path = os.path.join(
    project_root,
    "results",
    "tables",
    "signals",
    f"{symbol}_{factor_id}_{factor_name}_signals.csv"
)

summary_output_path = os.path.join(
    project_root,
    "results",
    "tables",
    "summary",
    f"{symbol}_{factor_id}_{factor_name}_summary.csv"
)

figures_dir = os.path.join(project_root, "results", "figures")

price_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_{factor_id}_{factor_name}_signal_on_price.png"
)

price_diff_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_{factor_id}_{factor_name}_price_and_diff.png"
)


# =========================
# 1. 读取已经做好的趋势数据
# =========================

daily = pd.read_csv(daily_input_path)
segments = pd.read_csv(segments_input_path)

daily["date"] = pd.to_datetime(daily["date"])
segments["start_date"] = pd.to_datetime(segments["start_date"])
segments["end_date"] = pd.to_datetime(segments["end_date"])

if "end_signal_date" in segments.columns:
    segments["end_signal_date"] = pd.to_datetime(segments["end_signal_date"])

if "reversal_start_date" in segments.columns:
    segments["reversal_start_date"] = pd.to_datetime(
        segments["reversal_start_date"]
    )

if "reversal_end_date" in segments.columns:
    segments["reversal_end_date"] = pd.to_datetime(
        segments["reversal_end_date"]
    )

daily = daily.sort_values("date").reset_index(drop=True)


# =========================
# 2. 检查是否已经有趋势反转段标签
# =========================
# is_reversal_window 应该来自 prepare_data.py 里的 add_reversal_window()
# is_reversal_window = 1 表示当天处于趋势结束点前3天到后2天的反转段

if "is_reversal_window" not in daily.columns:
    daily["is_reversal_window"] = 0

if "reversal_segment_id" not in daily.columns:
    daily["reversal_segment_id"] = np.nan


# =========================
# 3. 计算投机度一阶差分
# =========================
# DeltaSpec_t = Spec_t - Spec_{t-1}
# DeltaSpec_t > 0 表示投机度上升
# DeltaSpec_t < 0 表示投机度回落
# AbsDeltaSpec_t = |DeltaSpec_t| 表示投机行为变化幅度

daily["speculation_first_diff"] = (
    daily["speculation"] - daily["speculation"].shift(diff_days)
)

daily["speculation_abs_first_diff"] = (
    daily["speculation_first_diff"].abs()
)


# =========================
# 4. 判断一阶差分是否处于历史异常位置
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

    # 只用今天以前的数据
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
# 5. 生成标准化因子每日表
# =========================
# 为了和 11high_speculation.py 的接口一致：
# factor_value = 绝对一阶差分的历史分位数
# signal = 绝对一阶差分处于历史高位
#
# 这样这个因子表示：
# 投机度是否发生异常剧烈变化。

daily["factor_id"] = factor_id
daily["factor_name"] = factor_name

daily["factor_value"] = daily["speculation_abs_first_diff_rank"]

daily["signal"] = daily["speculation_first_diff_large_abs_signal"]

daily["position_scale"] = 1.0
daily.loc[daily["signal"] == 1, "position_scale"] = signal_position_scale

daily["is_effective_signal"] = (
    (daily["signal"] == 1) &
    (daily["is_reversal_window"] == 1)
).astype(int)


factor_daily = daily[
    [
        "date",
        "close",
        "trend",
        "segment_id",
        "is_reversal_window",
        "reversal_segment_id",

        "factor_id",
        "factor_name",
        "factor_value",

        "speculation",
        "speculation_first_diff",
        "speculation_abs_first_diff",
        "speculation_first_diff_rank",
        "speculation_abs_first_diff_rank",

        "speculation_first_diff_rise_signal",
        "speculation_first_diff_drop_signal",
        "speculation_first_diff_large_abs_signal",

        "signal",
        "position_scale",
        "is_effective_signal",
    ]
].copy()


# =========================
# 6. 生成标准化信号事件表
# =========================
# 只保留 signal == 1 的日期。
# 之后单因子分析可以看：
# 1. 信号出现在哪段趋势；
# 2. 是否落在趋势反转段；
# 3. 距离趋势结束点有多少天。

signal_points = daily[daily["signal"] == 1].copy()

signal_rows = []

for _, row in signal_points.iterrows():

    segment_id = row["segment_id"]

    if pd.isna(segment_id):
        continue

    matched_segment = segments[segments["segment_id"] == segment_id]

    if len(matched_segment) == 0:
        continue

    seg = matched_segment.iloc[0]

    signal_date = row["date"]
    end_date = seg["end_date"]

    days_to_trend_end = (end_date - signal_date).days

    signal_rows.append({
        "factor_id": factor_id,
        "factor_name": factor_name,

        "segment_id": segment_id,
        "trend": row["trend"],

        "signal_date": signal_date,
        "signal_close": row["close"],

        "factor_value": row["factor_value"],

        "speculation": row["speculation"],
        "speculation_first_diff": row["speculation_first_diff"],
        "speculation_abs_first_diff": row["speculation_abs_first_diff"],
        "speculation_first_diff_rank": row[
            "speculation_first_diff_rank"
        ],
        "speculation_abs_first_diff_rank": row[
            "speculation_abs_first_diff_rank"
        ],

        "rise_signal": row["speculation_first_diff_rise_signal"],
        "drop_signal": row["speculation_first_diff_drop_signal"],
        "large_abs_signal": row[
            "speculation_first_diff_large_abs_signal"
        ],

        "position_scale": row["position_scale"],

        "is_reversal_window": row["is_reversal_window"],
        "is_effective_signal": row["is_effective_signal"],

        "end_date": seg["end_date"],
        "end_close": seg["end_close"],

        "end_signal_date": seg.get("end_signal_date", pd.NaT),
        "end_signal_close": seg.get("end_signal_close", np.nan),

        "reversal_start_date": seg.get("reversal_start_date", pd.NaT),
        "reversal_end_date": seg.get("reversal_end_date", pd.NaT),

        "days_to_trend_end": days_to_trend_end,
    })

signal_table = pd.DataFrame(signal_rows)


# =========================
# 7. 按趋势段汇总
# =========================
# 这里不再用 before_end_days。
# 统一改成使用 prepare_data.py 里定义好的反转段：
# end_index 前3天到后2天。

trend_daily = daily[daily["trend"] != "no_trend"].copy()

result_rows = []

for _, seg in segments.iterrows():

    segment_id = seg["segment_id"]

    part = trend_daily[trend_daily["segment_id"] == segment_id].copy()

    if len(part) == 0:
        continue

    reversal_part = daily[
        daily["reversal_segment_id"] == segment_id
    ].copy()

    signal_part = part[part["signal"] == 1].copy()
    effective_signal_part = reversal_part[
        reversal_part["is_effective_signal"] == 1
    ].copy()

    if len(signal_part) > 0:
        first_signal_date = signal_part["date"].min()
        first_signal_close = signal_part.loc[
            signal_part["date"].idxmin(), "close"
        ]
        days_to_end_first_signal = (
            seg["end_date"] - first_signal_date
        ).days
    else:
        first_signal_date = pd.NaT
        first_signal_close = np.nan
        days_to_end_first_signal = np.nan

    if len(effective_signal_part) > 0:
        first_effective_signal_date = effective_signal_part["date"].min()
        first_effective_signal_close = effective_signal_part.loc[
            effective_signal_part["date"].idxmin(), "close"
        ]
        days_to_end_first_effective_signal = (
            seg["end_date"] - first_effective_signal_date
        ).days
    else:
        first_effective_signal_date = pd.NaT
        first_effective_signal_close = np.nan
        days_to_end_first_effective_signal = np.nan

    result_rows.append({
        "factor_id": factor_id,
        "factor_name": factor_name,

        "segment_id": segment_id,
        "trend": seg["trend"],

        "start_date": seg["start_date"],
        "end_date": seg["end_date"],
        "end_signal_date": seg.get("end_signal_date", pd.NaT),

        "reversal_start_date": seg.get("reversal_start_date", pd.NaT),
        "reversal_end_date": seg.get("reversal_end_date", pd.NaT),

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

        "rise_signal_days_in_trend": part[
            "speculation_first_diff_rise_signal"
        ].sum(),
        "drop_signal_days_in_trend": part[
            "speculation_first_diff_drop_signal"
        ].sum(),
        "large_abs_signal_days_in_trend": part[
            "speculation_first_diff_large_abs_signal"
        ].sum(),

        "signal_days_in_trend": part["signal"].sum(),
        "signal_ratio_in_trend": part["signal"].mean(),

        "rise_signal_days_in_reversal_window": reversal_part[
            "speculation_first_diff_rise_signal"
        ].sum(),
        "drop_signal_days_in_reversal_window": reversal_part[
            "speculation_first_diff_drop_signal"
        ].sum(),
        "large_abs_signal_days_in_reversal_window": reversal_part[
            "speculation_first_diff_large_abs_signal"
        ].sum(),

        "signal_days_in_reversal_window": reversal_part["signal"].sum(),
        "signal_ratio_in_reversal_window": reversal_part["signal"].mean(),

        "has_signal": int(part["signal"].sum() > 0),
        "has_effective_signal": int(
            reversal_part["is_effective_signal"].sum() > 0
        ),

        "first_signal_date": first_signal_date,
        "first_signal_close": first_signal_close,
        "days_to_end_first_signal": days_to_end_first_signal,

        "first_effective_signal_date": first_effective_signal_date,
        "first_effective_signal_close": first_effective_signal_close,
        "days_to_end_first_effective_signal": (
            days_to_end_first_effective_signal
        ),

        "mean_first_diff_in_reversal_window": reversal_part[
            "speculation_first_diff"
        ].mean(),
        "max_first_diff_in_reversal_window": reversal_part[
            "speculation_first_diff"
        ].max(),
        "min_first_diff_in_reversal_window": reversal_part[
            "speculation_first_diff"
        ].min(),
        "mean_abs_first_diff_in_reversal_window": reversal_part[
            "speculation_abs_first_diff"
        ].mean(),
        "max_abs_first_diff_in_reversal_window": reversal_part[
            "speculation_abs_first_diff"
        ].max(),

        "max_factor_value_in_reversal_window": reversal_part[
            "factor_value"
        ].max(),
        "mean_factor_value_in_reversal_window": reversal_part[
            "factor_value"
        ].mean(),
    })

first_diff_summary = pd.DataFrame(result_rows)


# =========================
# 8. 保存结果表格
# =========================

os.makedirs(os.path.dirname(factor_output_path), exist_ok=True)
os.makedirs(os.path.dirname(signal_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

factor_daily.to_csv(factor_output_path, index=False)
signal_table.to_csv(signal_output_path, index=False)
first_diff_summary.to_csv(summary_output_path, index=False)

print("因子 22：投机度一阶差分因子分析完成。")
print(f"因子每日值表保存为：{factor_output_path}")
print(f"因子信号事件表保存为：{signal_output_path}")
print(f"趋势段汇总表保存为：{summary_output_path}")

print("\n趋势段汇总预览：")
print(first_diff_summary.head(20))


# =========================
# 9. 画图：价格和投机度一阶差分信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily["close"], label="close", color="tab:blue")

rise_points = daily[daily["speculation_first_diff_rise_signal"] == 1]
drop_points = daily[daily["speculation_first_diff_drop_signal"] == 1]
large_abs_points = daily[daily["signal"] == 1]
effective_points = daily[daily["is_effective_signal"] == 1]

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

plt.scatter(
    large_abs_points["date"],
    large_abs_points["close"],
    s=signal_point_size * 1.5,
    marker="o",
    facecolors="none",
    edgecolors="black",
    label="large abs DeltaSpec signal"
)

plt.scatter(
    effective_points["date"],
    effective_points["close"],
    s=signal_point_size * 2,
    marker="X",
    color="tab:orange",
    label="effective signal in reversal window"
)

plt.title(f"{symbol} Close Price and Factor 22 DeltaSpec Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 10. 画图：价格和投机度一阶差分
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
