import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# 参数设置
# =========================

symbol = "LC"

factor_id = "11"
factor_name = "high_speculation"

history_window = 10  # 看过去多少个交易日作为“历史参照”
history_rank_threshold = 0.95  # 历史分位数阈值
trend_rank_threshold = 0.6  # 趋势分位数阈值
min_history_days = 10  # 最小历史天数
min_trend_rank_days = 3  # 最小趋势分位数天数

# 单因子触发后，建议仓位比例
signal_position_scale = 0.5

price_figsize = (12, 6)
rank_figsize = (12, 5)
signal_point_size = 20
plot_dpi = 300

# =========================
# 输入路径
# =========================

daily_input_path = f"../../results/tables/daily/{symbol}_daily_with_trend.csv"
segments_input_path = f"../../results/tables/daily/{symbol}_trend_segments.csv"

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

rank_figure_path = (
    f"../../results/figures/{symbol}_{factor_id}_{factor_name}_rank.png"
)

price_speculation_figure_path = (
    f"../../results/figures/{symbol}_{factor_id}_{factor_name}_price_and_speculation.png"
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
# 这个标签应该来自 prepare_data.py 里的 add_reversal_window()
# is_reversal_window = 1 表示：
# 当前日期处于趋势结束点前3天到后2天的趋势反转段内

if "is_reversal_window" not in daily.columns:
    daily["is_reversal_window"] = 0

if "reversal_segment_id" not in daily.columns:
    daily["reversal_segment_id"] = np.nan


# =========================
# 3. 计算历史和趋势段内的投机度分位数
# =========================
# 同时满足：
# 1. 今天投机度处于过去 history_window 天的历史高位；
# 2. 今天投机度处于当前趋势段开始以来的段内高位。
#
# 两个分位数都只使用“今天以前”的数据，避免未来数据泄露。

daily["speculation_rank_history"] = np.nan
daily["speculation_rank_in_trend"] = np.nan
daily["high_speculation_signal"] = 0


# =========================
# 3.1 计算历史分位数
# =========================

for i in range(len(daily)):

    current_speculation = daily.iloc[i]["speculation"]

    start_i = max(0, i - history_window)

    # 只使用今天以前的数据
    history = daily.iloc[start_i:i]["speculation"]

    if len(history) < min_history_days:
        continue

    daily.loc[daily.index[i], "speculation_rank_history"] = (
        history <= current_speculation
    ).mean()


# =========================
# 3.2 计算趋势段内分位数
# =========================

for _, seg in segments.iterrows():

    segment_id = seg["segment_id"]

    mask = daily["segment_id"] == segment_id

    part = daily[mask].copy()

    if len(part) == 0:
        continue

    ranks = []

    for i in range(len(part)):

        current_speculation = part.iloc[i]["speculation"]

        # 只使用趋势开始到昨天的数据
        history = part.iloc[:i]["speculation"]

        if len(history) < min_trend_rank_days:
            ranks.append(np.nan)
            continue

        rank = (history <= current_speculation).mean()

        ranks.append(rank)

    daily.loc[mask, "speculation_rank_in_trend"] = ranks


# =========================
# 3.3 生成高投机度信号
# =========================
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
# 4. 生成标准化因子每日表
# =========================
# 这个表是之后单因子回测、多因子回测的统一输入格式

daily["factor_id"] = factor_id
daily["factor_name"] = factor_name

# 这里选择趋势段内投机度分位数作为主因子值
daily["factor_value"] = daily["speculation_rank_in_trend"]

# 标准信号列
daily["signal"] = daily["high_speculation_signal"]

# 触发信号后建议仓位比例
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


# =========================
# 5. 输出标准化因子每日值表
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
        "speculation_rank_history",
        "speculation_rank_in_trend",

        "signal",
        "position_scale",
        "is_effective_signal",
    ]
].copy()


# =========================
# 6. 输出标准化信号事件表
# =========================
# 这个表只保留 signal == 1 的日期。
# 之后可以用它检查：
# 1. 哪些趋势段触发了信号；
# 2. 信号是否落在趋势反转段；
# 3. 信号距离趋势结束点有多少天。

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
        "speculation_rank_history": row["speculation_rank_history"],
        "speculation_rank_in_trend": row["speculation_rank_in_trend"],

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
# 这个表用来评价：
# 1. 每段趋势中有没有触发高投机度信号；
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

        # 整个趋势段内的高投机度信号
        "signal_days_in_trend": part["signal"].sum(),
        "signal_ratio_in_trend": part["signal"].mean(),

        # 趋势反转段内的高投机度信号
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

        # 反转段内最高/平均因子值
        "max_factor_value_in_reversal_window": (
            reversal_part["factor_value"].max()
        ),
        "mean_factor_value_in_reversal_window": (
            reversal_part["factor_value"].mean()
        ),

        # 反转段内最高/平均历史分位数
        "max_history_rank_in_reversal_window": (
            reversal_part["speculation_rank_history"].max()
        ),
        "mean_history_rank_in_reversal_window": (
            reversal_part["speculation_rank_history"].mean()
        ),
    })

high_spec_summary = pd.DataFrame(result_rows)


# =========================
# 8. 保存结果表格
# =========================

os.makedirs(os.path.dirname(factor_output_path), exist_ok=True)
os.makedirs(os.path.dirname(signal_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

factor_daily.to_csv(factor_output_path, index=False)
signal_table.to_csv(signal_output_path, index=False)
high_spec_summary.to_csv(summary_output_path, index=False)

print("因子 11：高投机度分析完成。")
print(f"因子每日值表保存为：{factor_output_path}")
print(f"因子信号事件表保存为：{signal_output_path}")
print(f"趋势段汇总表保存为：{summary_output_path}")

print("\n趋势段汇总预览：")
print(high_spec_summary.head(20))


# =========================
# 9. 画图：价格、高投机度信号、有效信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily["close"], label="close")

signal_points = daily[daily["signal"] == 1]
effective_points = daily[daily["is_effective_signal"] == 1]

plt.scatter(
    signal_points["date"],
    signal_points["close"],
    s=signal_point_size,
    label="high speculation signal"
)

plt.scatter(
    effective_points["date"],
    effective_points["close"],
    s=signal_point_size * 2,
    marker="X",
    label="effective signal in reversal window"
)

plt.title(f"{symbol} Close Price and Factor 11 High Speculation Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 10. 画图：价格和投机度
# =========================

fig, ax1 = plt.subplots(figsize=price_figsize)

ax1.plot(daily["date"], daily["close"], label="close", color="tab:blue")
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(
    daily["date"],
    daily["speculation"],
    label="speculation",
    color="tab:orange"
)
ax2.set_ylabel("Speculation", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

plt.title(f"{symbol} Close Price and Speculation")
fig.tight_layout()

plt.savefig(price_speculation_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 11. 画图：趋势段内投机度分位数
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

plt.title("Factor 11: Speculation Rank Within Trend and History")
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
print(price_speculation_figure_path)
