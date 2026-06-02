import os

# 当前文件在 code/factors/ 下面
# project_root 指向项目根目录 LcResearch/
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

factor_id = "23"
factor_name = "speculation_continuous_drop"

diff_days = 1  # 和多少个交易日前相比，计算投机度一阶差分
down_window = 5  # 过去多少个交易日内统计投机度下降天数
down_days_threshold = 4  # 过去 down_window 天里，投机度下降天数达到该值则触发信号
min_down_window_days = 3  # 统计下降天数所需的最少历史天数
strictly_negative_diff = True  # True 表示 DeltaSpec < 0 才算下降

# 单因子触发后，建议仓位比例
signal_position_scale = 0.5

price_figsize = (12, 6)
down_days_figsize = (12, 6)
signal_point_size = 22
plot_dpi = 300

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

# 如果 prepare_data.py 还没有改成 daily 文件夹，则自动读取旧路径
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

price_down_days_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_{factor_id}_{factor_name}_price_and_down_days.png"
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
# 趋势结束点前3天 + 结束当天 + 后2天，属于趋势反转段

if "is_reversal_window" not in daily.columns:
    daily["is_reversal_window"] = 0

if "reversal_segment_id" not in daily.columns:
    daily["reversal_segment_id"] = np.nan


# =========================
# 3. 计算投机度连续回落因子
# =========================
# DeltaSpec_t = Spec_t - Spec_{t-1}
# SpecDownDays_t = sum(I(DeltaSpec_i < 0)), i from t-k+1 to t

daily["speculation_first_diff"] = (
    daily["speculation"] - daily["speculation"].shift(diff_days)
)

if strictly_negative_diff:
    daily["speculation_down_flag"] = (
        daily["speculation_first_diff"] < 0
    ).astype(float)
else:
    daily["speculation_down_flag"] = (
        daily["speculation_first_diff"] <= 0
    ).astype(float)

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
# 4. 生成标准化因子每日表
# =========================
# 这个表是之后单因子回测、多因子回测的统一输入格式

daily["factor_id"] = factor_id
daily["factor_name"] = factor_name

# 主因子值：过去 down_window 天里，投机度下降的天数
daily["factor_value"] = daily["speculation_down_days"]

# 标准信号列
daily["signal"] = daily["speculation_continuous_drop_signal"]

# 没有信号：仓位比例 1.0
# 有信号：仓位比例 signal_position_scale
daily["position_scale"] = 1.0
daily.loc[daily["signal"] == 1, "position_scale"] = signal_position_scale

# 是否为有效信号：
# 只有在趋势反转段内触发的信号，才算有效
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
        "speculation_down_flag",
        "speculation_down_days",
        "speculation_down_streak",

        "signal",
        "position_scale",
        "is_effective_signal",
    ]
].copy()


# =========================
# 5. 生成标准化信号事件表
# =========================
# 只保留 signal == 1 的日期

signal_points = daily[daily["signal"] == 1].copy()

signal_rows = []

for _, row in signal_points.iterrows():

    segment_id = row["segment_id"]

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
        "speculation_down_days": row["speculation_down_days"],
        "speculation_down_streak": row["speculation_down_streak"],

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
# 6. 按趋势段汇总
# =========================
# 用来评价：
# 1. 每段趋势中有没有触发连续回落信号；
# 2. 有没有在趋势反转段内触发；
# 3. 有效信号出现了多少次。

trend_daily = daily[daily["trend"] != "no_trend"].copy()

result_rows = []

for _, seg in segments.iterrows():

    segment_id = seg["segment_id"]

    part = trend_daily[trend_daily["segment_id"] == segment_id].copy()

    if len(part) == 0:
        continue

    # 当前趋势段对应的趋势反转段
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
            signal_part["date"].idxmin(),
            "close"
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
            effective_signal_part["date"].idxmin(),
            "close"
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

        # 整个趋势段内的因子表现
        "mean_factor_value_in_trend": part["factor_value"].mean(),
        "max_factor_value_in_trend": part["factor_value"].max(),

        "mean_down_streak_in_trend": part[
            "speculation_down_streak"
        ].mean(),
        "max_down_streak_in_trend": part[
            "speculation_down_streak"
        ].max(),

        "signal_days_in_trend": part["signal"].sum(),
        "signal_ratio_in_trend": part["signal"].mean(),

        # 反转段内的因子表现
        "mean_factor_value_in_reversal_window": (
            reversal_part["factor_value"].mean()
        ),
        "max_factor_value_in_reversal_window": (
            reversal_part["factor_value"].max()
        ),

        "mean_down_streak_in_reversal_window": (
            reversal_part["speculation_down_streak"].mean()
        ),
        "max_down_streak_in_reversal_window": (
            reversal_part["speculation_down_streak"].max()
        ),

        "signal_days_in_reversal_window": reversal_part["signal"].sum(),
        "signal_ratio_in_reversal_window": reversal_part["signal"].mean(),

        # 是否有信号
        "has_signal": int(part["signal"].sum() > 0),

        # 是否有有效信号
        "has_effective_signal": int(
            reversal_part["is_effective_signal"].sum() > 0
        ),

        # 第一次信号
        "first_signal_date": first_signal_date,
        "first_signal_close": first_signal_close,
        "days_to_end_first_signal": days_to_end_first_signal,

        # 第一次有效信号
        "first_effective_signal_date": first_effective_signal_date,
        "first_effective_signal_close": first_effective_signal_close,
        "days_to_end_first_effective_signal": (
            days_to_end_first_effective_signal
        ),
    })

continuous_drop_summary = pd.DataFrame(result_rows)


# =========================
# 7. 保存结果表格
# =========================

os.makedirs(os.path.dirname(factor_output_path), exist_ok=True)
os.makedirs(os.path.dirname(signal_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

factor_daily.to_csv(factor_output_path, index=False)
signal_table.to_csv(signal_output_path, index=False)
continuous_drop_summary.to_csv(summary_output_path, index=False)

print("因子 23：投机度连续回落分析完成。")
print(f"因子每日值表保存为：{factor_output_path}")
print(f"因子信号事件表保存为：{signal_output_path}")
print(f"趋势段汇总表保存为：{summary_output_path}")

print("\n趋势段汇总预览：")
print(continuous_drop_summary.head(20))


# =========================
# 8. 画图：价格和投机度连续回落信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily["close"], label="close", color="tab:blue")

signal_points = daily[daily["signal"] == 1]
effective_points = daily[daily["is_effective_signal"] == 1]

plt.scatter(
    signal_points["date"],
    signal_points["close"],
    s=signal_point_size,
    label=f"SpecDownDays >= {down_days_threshold}",
    color="tab:green"
)

plt.scatter(
    effective_points["date"],
    effective_points["close"],
    s=signal_point_size * 2,
    marker="X",
    label="effective signal in reversal window",
    color="tab:red"
)

plt.title(f"{symbol} Close Price and Factor 23 Continuous Drop Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 9. 画图：价格和投机度连续回落天数
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

plt.title(f"{symbol} Close Price and Factor 23 Continuous Drop")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
fig.tight_layout()

os.makedirs(os.path.dirname(price_down_days_figure_path), exist_ok=True)
plt.savefig(price_down_days_figure_path, dpi=plot_dpi)
plt.close()

print("\n图片保存为：")
print(price_figure_path)
print(price_down_days_figure_path)
