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

factor_id = "21"
factor_name = "speculation_change_rate"

change_days = 1  # 和多少个交易日前相比，计算投机度变化率
history_window = 20  # 用过去多少个交易日判断变化率是否异常
high_change_rank_threshold = 0.99  # 变化率处于历史高位的分位数阈值
low_change_rank_threshold = 0.01  # 变化率处于历史低位的分位数阈值
min_history_days = 10  # 计算历史分位数所需的最少历史天数

# 单因子触发后，建议仓位比例
signal_position_scale = 0.5

price_figsize = (12, 6)
change_figsize = (12, 5)
signal_point_size = 20
plot_dpi = 300

# =========================
# 输入路径
# =========================
# 当前文件在 code/factors/ 下面运行，所以要用 ../../ 回到项目根目录

daily_input_path = f"../../results/tables/daily/{symbol}_daily_with_trend.csv"
segments_input_path = f"../../results/tables/daily/{symbol}_trend_segments.csv"

# 如果 prepare_data.py 还没有改成 daily 文件夹，则自动读取旧路径
if not os.path.exists(daily_input_path):
    daily_input_path = f"../../results/tables/{symbol}_daily_with_trend.csv"

if not os.path.exists(segments_input_path):
    segments_input_path = f"../../results/tables/{symbol}_trend_segments.csv"


# =========================
# 输出路径
# =========================

factor_output_path = (
    f"../../results/tables/factors/{symbol}_{factor_id}_{factor_name}.csv"
)

signal_output_path = (
    f"../../results/tables/signals/{symbol}_{factor_id}_{factor_name}_signals.csv"
)

summary_output_path = (
    f"../../results/tables/summary/{symbol}_{factor_id}_{factor_name}_summary.csv"
)

price_figure_path = (
    f"../../results/figures/{symbol}_{factor_id}_{factor_name}_signal_on_price.png"
)

change_figure_path = (
    f"../../results/figures/{symbol}_{factor_id}_{factor_name}.png"
)

price_change_figure_path = (
    f"../../results/figures/{symbol}_{factor_id}_{factor_name}_price_and_factor.png"
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


# =========================
# 2. 检查是否已经有趋势反转段标签
# =========================
# is_reversal_window = 1 表示：
# 当前日期处于趋势结束点前3天到后2天的趋势反转段内

if "is_reversal_window" not in daily.columns:
    daily["is_reversal_window"] = 0

if "reversal_segment_id" not in daily.columns:
    daily["reversal_segment_id"] = np.nan


# =========================
# 3. 计算投机度变化率
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
# 4. 判断变化率是否处于历史高位或低位
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

    # 只使用今天以前的数据
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
# 5. 生成标准化因子每日表
# =========================
# 这个表是之后单因子回测、多因子回测的统一输入格式。

daily["factor_id"] = factor_id
daily["factor_name"] = factor_name

# 主因子值：投机度变化率
daily["factor_value"] = daily["speculation_change_rate"]

# 统一信号列：
# 投机度变化率极端上升或极端下降，都认为是一个变化率异常信号
daily["signal"] = (
    (daily["speculation_fast_rise_signal"] == 1) |
    (daily["speculation_fast_drop_signal"] == 1)
).astype(int)

# 信号方向：
# fast_rise 表示投机度快速上升；
# fast_drop 表示投机度快速下降；
# none 表示没有信号。
daily["signal_type"] = "none"

daily.loc[
    daily["speculation_fast_rise_signal"] == 1,
    "signal_type"
] = "fast_rise"

daily.loc[
    daily["speculation_fast_drop_signal"] == 1,
    "signal_type"
] = "fast_drop"

# 理论上同一天不太会同时处于 0.98 和 0.02 分位数；
# 这里保留一个兜底。
daily.loc[
    (
        daily["speculation_fast_rise_signal"] == 1
    ) & (
        daily["speculation_fast_drop_signal"] == 1
    ),
    "signal_type"
] = "both"

# 触发信号后建议仓位比例
daily["position_scale"] = 1.0
daily.loc[daily["signal"] == 1, "position_scale"] = signal_position_scale

# 是否为有效信号：
# 只有在趋势反转段内触发的信号，才算有效
daily["is_effective_signal"] = (
    (daily["signal"] == 1) &
    (daily["is_reversal_window"] == 1)
).astype(int)


# =========================
# 6. 输出标准化因子每日值表
# =========================

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
        "speculation_change",
        "speculation_change_rate",
        "speculation_change_rate_rank",

        "speculation_fast_rise_signal",
        "speculation_fast_drop_signal",
        "signal_type",

        "signal",
        "position_scale",
        "is_effective_signal",
    ]
].copy()


# =========================
# 7. 输出标准化信号事件表
# =========================
# 这个表只保留 signal == 1 的日期。
# 用来检查信号是否落在趋势反转段内。

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
        "speculation_change": row["speculation_change"],
        "speculation_change_rate": row["speculation_change_rate"],
        "speculation_change_rate_rank": (
            row["speculation_change_rate_rank"]
        ),

        "speculation_fast_rise_signal": (
            row["speculation_fast_rise_signal"]
        ),
        "speculation_fast_drop_signal": (
            row["speculation_fast_drop_signal"]
        ),
        "signal_type": row["signal_type"],

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
# 8. 按趋势段汇总
# =========================
# 这个表用来评价：
# 1. 每段趋势中有没有触发投机度变化率异常信号；
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

        # 整个趋势段内的变化率统计
        "mean_change_rate_in_trend": (
            part["speculation_change_rate"].mean()
        ),
        "max_change_rate_in_trend": (
            part["speculation_change_rate"].max()
        ),
        "min_change_rate_in_trend": (
            part["speculation_change_rate"].min()
        ),

        # 整个趋势段内的信号
        "signal_days_in_trend": part["signal"].sum(),
        "signal_ratio_in_trend": part["signal"].mean(),

        "fast_rise_days_in_trend": (
            part["speculation_fast_rise_signal"].sum()
        ),
        "fast_rise_ratio_in_trend": (
            part["speculation_fast_rise_signal"].mean()
        ),
        "fast_drop_days_in_trend": (
            part["speculation_fast_drop_signal"].sum()
        ),
        "fast_drop_ratio_in_trend": (
            part["speculation_fast_drop_signal"].mean()
        ),

        # 趋势反转段内的变化率统计
        "mean_change_rate_in_reversal_window": (
            reversal_part["speculation_change_rate"].mean()
        ),
        "max_change_rate_in_reversal_window": (
            reversal_part["speculation_change_rate"].max()
        ),
        "min_change_rate_in_reversal_window": (
            reversal_part["speculation_change_rate"].min()
        ),

        # 趋势反转段内的信号
        "signal_days_in_reversal_window": (
            reversal_part["signal"].sum()
        ),
        "signal_ratio_in_reversal_window": (
            reversal_part["signal"].mean()
        ),

        "fast_rise_days_in_reversal_window": (
            reversal_part["speculation_fast_rise_signal"].sum()
        ),
        "fast_rise_ratio_in_reversal_window": (
            reversal_part["speculation_fast_rise_signal"].mean()
        ),
        "fast_drop_days_in_reversal_window": (
            reversal_part["speculation_fast_drop_signal"].sum()
        ),
        "fast_drop_ratio_in_reversal_window": (
            reversal_part["speculation_fast_drop_signal"].mean()
        ),

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

        # 反转段内最高/平均因子值
        "max_factor_value_in_reversal_window": (
            reversal_part["factor_value"].max()
        ),
        "mean_factor_value_in_reversal_window": (
            reversal_part["factor_value"].mean()
        ),

        # 反转段内最高/平均变化率分位数
        "max_rank_in_reversal_window": (
            reversal_part["speculation_change_rate_rank"].max()
        ),
        "mean_rank_in_reversal_window": (
            reversal_part["speculation_change_rate_rank"].mean()
        ),
    })

change_rate_summary = pd.DataFrame(result_rows)


# =========================
# 9. 保存结果表格
# =========================

os.makedirs(os.path.dirname(factor_output_path), exist_ok=True)
os.makedirs(os.path.dirname(signal_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

factor_daily.to_csv(factor_output_path, index=False)
signal_table.to_csv(signal_output_path, index=False)
change_rate_summary.to_csv(summary_output_path, index=False)

print("因子 21：投机度变化率分析完成。")
print(f"因子每日值表保存为：{factor_output_path}")
print(f"因子信号事件表保存为：{signal_output_path}")
print(f"趋势段汇总表保存为：{summary_output_path}")

print("\n趋势段汇总预览：")
print(change_rate_summary.head(20))


# =========================
# 10. 画图：价格和变化率信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily["close"], label="close")

fast_rise_points = daily[daily["speculation_fast_rise_signal"] == 1]
fast_drop_points = daily[daily["speculation_fast_drop_signal"] == 1]
effective_points = daily[daily["is_effective_signal"] == 1]

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

plt.scatter(
    effective_points["date"],
    effective_points["close"],
    s=signal_point_size * 2,
    marker="X",
    label="effective signal in reversal window"
)

plt.title(f"{symbol} Close Price and Factor 21 Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 11. 画图：价格和投机度变化率
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

plt.title(f"{symbol} Close Price and Factor 21 Speculation Change Rate")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
fig.tight_layout()

plt.savefig(price_change_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 12. 画图：投机度变化率
# =========================

plt.figure(figsize=change_figsize)

plt.plot(
    daily["date"],
    daily["speculation_change_rate"],
    label="speculation change rate"
)

plt.axhline(0, color="gray", linewidth=1)

plt.title("Factor 21: Speculation Change Rate")
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