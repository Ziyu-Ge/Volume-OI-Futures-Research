import os

# 注意：
# 当前文件在 code/factors/ 下面
# 所以 project_root 需要往上走两层，回到 LcResearch/
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(project_root, ".matplotlib")
)

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 参数设置
# =========================

symbol = "LC"

factor_id = "12"
factor_name = "speculation_mad"

history_window = 10  # 用过去多少个交易日计算 MAD
min_history_days = 10  # 计算 MAD 所需的最少历史天数

mad_threshold = 3  # MAD 标准化偏离值绝对值高于该阈值，认为投机度极端偏离
mad_scale = 1.4826  # 把 MAD 调整到类似标准差的尺度
mad_epsilon = 1e-12  # 避免 MAD 过小导致除零

# 单因子触发后，建议仓位比例
signal_position_scale = 0.7

price_figsize = (12, 6)
mad_figsize = (12, 6)
signal_point_size = 22
plot_dpi = 300


# =========================
# 输入路径
# =========================

tables_dir = os.path.join(project_root, "results", "tables")
figures_dir = os.path.join(project_root, "results", "figures")

daily_input_path = os.path.join(
    tables_dir,
    "daily",
    f"{symbol}_daily_with_trend.csv"
)

segments_input_path = os.path.join(
    tables_dir,
    "daily",
    f"{symbol}_trend_segments.csv"
)

# 如果 prepare_data.py 还没有输出到 daily 文件夹，则自动使用旧路径
if not os.path.exists(daily_input_path):
    daily_input_path = os.path.join(
        tables_dir,
        f"{symbol}_daily_with_trend.csv"
    )

if not os.path.exists(segments_input_path):
    segments_input_path = os.path.join(
        tables_dir,
        f"{symbol}_trend_segments.csv"
    )


# =========================
# 输出路径
# =========================

factor_output_path = os.path.join(
    tables_dir,
    "factors",
    f"{symbol}_{factor_id}_{factor_name}.csv"
)

signal_output_path = os.path.join(
    tables_dir,
    "signals",
    f"{symbol}_{factor_id}_{factor_name}_signals.csv"
)

summary_output_path = os.path.join(
    tables_dir,
    "summary",
    f"{symbol}_{factor_id}_{factor_name}_summary.csv"
)

price_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_{factor_id}_{factor_name}_signal_on_price.png"
)

price_mad_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_{factor_id}_{factor_name}_price_and_mad.png"
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
# is_reversal_window = 1 表示：
# 当前日期处于趋势结束点前3天到后2天的趋势反转段内

if "is_reversal_window" not in daily.columns:
    daily["is_reversal_window"] = 0

if "reversal_segment_id" not in daily.columns:
    daily["reversal_segment_id"] = np.nan


# =========================
# 3. 计算投机度 MAD 稳健偏离因子
# =========================
# 原来是：
# SpecZ_t = (Spec_t - mean) / std
#
# 现在改成：
# SpecMAD_t = (Spec_t - median) / (1.4826 * MAD)
#
# 其中：
# MAD = median(|Spec_i - median(Spec_i)|)
#
# 注意：
# median 和 MAD 都只使用今天以前过去 N 天的数据，
# 避免未来数据泄露。

daily["speculation_median_history"] = np.nan
daily["speculation_mad_history"] = np.nan
daily["speculation_mad_value"] = np.nan
daily["speculation_mad_signal"] = 0


for i in range(len(daily)):

    current_speculation = daily.loc[i, "speculation"]

    start_i = max(0, i - history_window)

    # 只使用今天以前的数据
    history = daily.loc[start_i:i - 1, "speculation"].dropna()

    if len(history) < min_history_days:
        continue

    history_median = history.median()

    history_mad = (history - history_median).abs().median()

    if abs(history_mad) <= mad_epsilon:
        continue

    mad_value = (
        (current_speculation - history_median)
        / (mad_scale * history_mad)
    )

    daily.loc[i, "speculation_median_history"] = history_median
    daily.loc[i, "speculation_mad_history"] = history_mad
    daily.loc[i, "speculation_mad_value"] = mad_value


daily.loc[
    daily["speculation_mad_value"].abs() > mad_threshold,
    "speculation_mad_signal"
] = 1


# =========================
# 4. 生成标准化因子每日表
# =========================
# 这个表是之后单因子回测、多因子回测的统一输入格式

daily["factor_id"] = factor_id
daily["factor_name"] = factor_name

# 主因子值：MAD 标准化偏离值
daily["factor_value"] = daily["speculation_mad_value"]

# 标准信号列
daily["signal"] = daily["speculation_mad_signal"]

# 没有信号：仓位比例 1.0
# 有信号：仓位比例 signal_position_scale
daily["position_scale"] = 1.0
daily.loc[daily["signal"] == 1, "position_scale"] = signal_position_scale

# 是否为有效信号：
# 只有在趋势反转段内触发的信号，才算有效
daily["is_effective_signal"] = (
    (daily["signal"] == 1)
    & (daily["is_reversal_window"] == 1)
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
        "speculation_median_history",
        "speculation_mad_history",
        "speculation_mad_value",

        "signal",
        "position_scale",
        "is_effective_signal",
    ]
].copy()


# =========================
# 6. 输出标准化信号事件表
# =========================
# 这个表只记录 signal == 1 的日期。
# 如果信号落在趋势反转段内，则它对 reversal_segment_id 对应的趋势段有效。

signal_points = daily[daily["signal"] == 1].copy()

signal_rows = []

for _, row in signal_points.iterrows():

    current_segment_id = row["segment_id"]

    if row["is_effective_signal"] == 1:
        evaluation_segment_id = row["reversal_segment_id"]
    else:
        evaluation_segment_id = current_segment_id

    matched_segment = segments[
        segments["segment_id"] == evaluation_segment_id
    ]

    if len(matched_segment) == 0:
        continue

    seg = matched_segment.iloc[0]

    signal_date = row["date"]
    end_date = seg["end_date"]

    days_to_trend_end = (end_date - signal_date).days

    signal_rows.append({
        "factor_id": factor_id,
        "factor_name": factor_name,

        # 当前交易日所属趋势段
        "segment_id": current_segment_id,

        # 用于判断有效性的趋势段
        "evaluation_segment_id": evaluation_segment_id,

        "trend": row["trend"],

        "signal_date": signal_date,
        "signal_close": row["close"],

        "factor_value": row["factor_value"],
        "speculation": row["speculation"],
        "speculation_median_history": row["speculation_median_history"],
        "speculation_mad_history": row["speculation_mad_history"],
        "speculation_mad_value": row["speculation_mad_value"],

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
# 1. 每段趋势中有没有触发 MAD 极端偏离信号；
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

        # 整个趋势段内的 MAD 因子表现
        "mean_factor_value_in_trend": part["factor_value"].mean(),
        "max_factor_value_in_trend": part["factor_value"].max(),
        "mean_mad_in_trend": part["speculation_mad_history"].mean(),

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

        # 反转段内最高/平均因子值
        "max_factor_value_in_reversal_window": (
            reversal_part["factor_value"].max()
        ),
        "mean_factor_value_in_reversal_window": (
            reversal_part["factor_value"].mean()
        ),

        # 反转段内最高/平均 MAD
        "max_mad_value_in_reversal_window": (
            reversal_part["speculation_mad_value"].max()
        ),
        "mean_mad_value_in_reversal_window": (
            reversal_part["speculation_mad_value"].mean()
        ),
    })

mad_summary = pd.DataFrame(result_rows)


# =========================
# 8. 保存结果表格
# =========================

os.makedirs(os.path.dirname(factor_output_path), exist_ok=True)
os.makedirs(os.path.dirname(signal_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

factor_daily.to_csv(factor_output_path, index=False)
signal_table.to_csv(signal_output_path, index=False)
mad_summary.to_csv(summary_output_path, index=False)

print("因子 12：投机度 MAD 极端偏离分析完成。")
print(f"因子每日值表保存为：{factor_output_path}")
print(f"因子信号事件表保存为：{signal_output_path}")
print(f"趋势段汇总表保存为：{summary_output_path}")

print("\n趋势段汇总预览：")
print(mad_summary.head(20))


# =========================
# 9. 画图：价格和 MAD 极端偏离信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily["close"], label="close", color="tab:blue")

signal_points = daily[daily["signal"] == 1]
effective_points = daily[daily["is_effective_signal"] == 1]

plt.scatter(
    signal_points["date"],
    signal_points["close"],
    s=signal_point_size,
    label=f"|MAD signal| > {mad_threshold}",
    color="tab:red"
)

plt.scatter(
    effective_points["date"],
    effective_points["close"],
    s=signal_point_size * 2,
    marker="X",
    label="effective signal in reversal window",
    color="tab:green"
)

plt.title(f"{symbol} Close Price and Factor 12 Speculation MAD Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 10. 画图：价格和 MAD 标准化偏离值
# =========================

fig, ax1 = plt.subplots(figsize=mad_figsize)

ax1.plot(daily["date"], daily["close"], label="close", color="tab:blue")
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(
    daily["date"],
    daily["speculation_mad_value"],
    label="Speculation MAD value",
    color="tab:orange"
)

ax2.axhline(0, color="gray", linewidth=1)
ax2.axhline(
    mad_threshold,
    color="tab:red",
    linestyle="--",
    linewidth=1,
    label=f"threshold +/- {mad_threshold}"
)
ax2.axhline(
    -mad_threshold,
    color="tab:red",
    linestyle="--",
    linewidth=1
)

ax2.set_ylabel("Speculation MAD Value", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

plt.title(f"{symbol} Close Price and Speculation MAD Value")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
fig.tight_layout()

plt.savefig(price_mad_figure_path, dpi=plot_dpi)
plt.close()

print("\n图片保存为：")
print(price_figure_path)
print(price_mad_figure_path)
