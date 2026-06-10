import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 参数设置
# =========================

symbol = "LC"

factor_id = "41"
factor_name = "oi_change_rate"

# 持仓量变化率阈值
# 0 表示：只要持仓量比上一交易日下降，就触发信号
# 如果你想更严格，可以改成 -2，表示持仓量下降超过 2% 才触发
oi_change_rate_threshold = 0

# 单因子触发后，建议仓位比例
signal_position_scale = 0.7

price_figsize = (12, 6)
factor_figsize = (12, 5)
signal_point_size = 20
plot_dpi = 300


# =========================
# 输入路径
# =========================

daily_input_path = f"../../results/tables/daily/{symbol}_daily_with_trend.csv"
segments_input_path = f"../../results/tables/daily/{symbol}_trend_segments.csv"

# 如果 prepare_data.py 还没有输出到 daily 文件夹，则自动读取旧路径
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

factor_figure_path = (
    f"../../results/figures/{symbol}_{factor_id}_{factor_name}.png"
)

price_oi_figure_path = (
    f"../../results/figures/{symbol}_{factor_id}_{factor_name}_price_and_oi.png"
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
# 2. 检查是否有趋势反转段标签
# =========================
# is_reversal_window = 1 表示：
# 当前日期处于趋势结束点前3天到后2天的趋势反转段内

if "is_reversal_window" not in daily.columns:
    daily["is_reversal_window"] = 0

if "reversal_segment_id" not in daily.columns:
    daily["reversal_segment_id"] = np.nan


# =========================
# 3. 计算持仓量变化率因子
# =========================
# 持仓量变化率 = (当日持仓量 - 上一交易日持仓量) / 上一交易日持仓量 * 100%

daily["oi_change_rate"] = (
    (daily["open_interest"] - daily["open_interest"].shift(1))
    / daily["open_interest"].shift(1)
    * 100
)

# 避免上一日持仓量为 0 或缺失时出现无效值
daily.loc[
    daily["open_interest"].shift(1) <= 0,
    "oi_change_rate"
] = np.nan


# =========================
# 4. 生成持仓量下降信号
# =========================
# 这里的逻辑是：
# 如果持仓量变化率 <= 阈值，则认为持仓资金正在退出，触发减仓信号

daily["oi_change_rate_signal"] = 0

daily.loc[
    daily["oi_change_rate"] <= oi_change_rate_threshold,
    "oi_change_rate_signal"
] = 1

# 只在趋势段内触发信号
daily.loc[daily["trend"] == "no_trend", "oi_change_rate_signal"] = 0


# =========================
# 5. 生成标准化因子每日表
# =========================
# 这个表是之后单因子回测、多因子回测的统一输入格式

daily["factor_id"] = factor_id
daily["factor_name"] = factor_name

# 主因子值：持仓量变化率，单位是 %
daily["factor_value"] = daily["oi_change_rate"]

# 标准信号列
daily["signal"] = daily["oi_change_rate_signal"]

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

        "open_interest",
        "oi_change_rate",

        "signal",
        "position_scale",
        "is_effective_signal",
    ]
].copy()


# =========================
# 7. 输出标准化信号事件表
# =========================
# 这个表只保留 signal == 1 的日期

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
        "open_interest": row["open_interest"],
        "oi_change_rate": row["oi_change_rate"],

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
# 1. 每段趋势中有没有触发持仓量下降信号；
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

        # 整个趋势段内的信号
        "signal_days_in_trend": part["signal"].sum(),
        "signal_ratio_in_trend": part["signal"].mean(),

        # 趋势反转段内的信号
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

        # 因子值统计
        "min_factor_value_in_trend": part["factor_value"].min(),
        "mean_factor_value_in_trend": part["factor_value"].mean(),

        "min_factor_value_in_reversal_window": (
            reversal_part["factor_value"].min()
        ),
        "mean_factor_value_in_reversal_window": (
            reversal_part["factor_value"].mean()
        ),

        # 持仓量统计
        "start_open_interest": part["open_interest"].iloc[0],
        "end_open_interest": part["open_interest"].iloc[-1],
        "mean_open_interest": part["open_interest"].mean(),
    })

oi_change_summary = pd.DataFrame(result_rows)


# =========================
# 9. 保存结果表格
# =========================

os.makedirs(os.path.dirname(factor_output_path), exist_ok=True)
os.makedirs(os.path.dirname(signal_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

factor_daily.to_csv(factor_output_path, index=False)
signal_table.to_csv(signal_output_path, index=False)
oi_change_summary.to_csv(summary_output_path, index=False)

print("因子 41：持仓量变化率分析完成。")
print(f"因子每日值表保存为：{factor_output_path}")
print(f"因子信号事件表保存为：{signal_output_path}")
print(f"趋势段汇总表保存为：{summary_output_path}")

print("\n趋势段汇总预览：")
print(oi_change_summary.head(20))


# =========================
# 10. 画图：价格、持仓量下降信号、有效信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily["close"], label="close")

signal_points = daily[daily["signal"] == 1]
effective_points = daily[daily["is_effective_signal"] == 1]

plt.scatter(
    signal_points["date"],
    signal_points["close"],
    s=signal_point_size,
    label="OI decrease signal"
)

plt.scatter(
    effective_points["date"],
    effective_points["close"],
    s=signal_point_size * 2,
    marker="X",
    label="effective signal in reversal window"
)

plt.title(f"{symbol} Close Price and Factor 41 OI Change Rate Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 11. 画图：持仓量变化率
# =========================

plt.figure(figsize=factor_figsize)

plt.plot(
    daily["date"],
    daily["oi_change_rate"],
    label="OI change rate (%)"
)

plt.axhline(
    oi_change_rate_threshold,
    linestyle="--",
    label="signal threshold"
)

plt.title(f"{symbol} Factor 41: Open Interest Change Rate")
plt.xlabel("Date")
plt.ylabel("OI Change Rate (%)")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(factor_figure_path), exist_ok=True)
plt.savefig(factor_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 12. 画图：价格和持仓量
# =========================

fig, ax1 = plt.subplots(figsize=price_figsize)

ax1.plot(daily["date"], daily["close"], label="close", color="tab:blue")
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(
    daily["date"],
    daily["open_interest"],
    label="open interest",
    color="tab:orange"
)
ax2.set_ylabel("Open Interest", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

plt.title(f"{symbol} Close Price and Open Interest")
fig.tight_layout()

os.makedirs(os.path.dirname(price_oi_figure_path), exist_ok=True)
plt.savefig(price_oi_figure_path, dpi=plot_dpi)
plt.close()


print("\n图片保存为：")
print(price_figure_path)
print(factor_figure_path)
print(price_oi_figure_path)
