import os

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(project_root, ".matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(project_root, ".cache")
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

factor_id = "42"
factor_name = "price_up_oi_down"

# 过去多少天判断“价格大部分时间上涨”
price_window = 10

# 过去多少天作为持仓量历史参照
oi_window = 10

# 过去 price_window 天中，至少多少比例是上涨日
price_up_ratio_threshold = 0.7

# 持仓量明显低于过去一段时间的阈值
# OI MAD score <= -1 表示今天持仓量明显低于过去一段时间
oi_mad_threshold = -1

# MAD 缩放系数，把 MAD 调整到类似标准差的尺度
mad_scale = 1.4826

# 避免 MAD 为 0
mad_epsilon = 1e-12

# 最小历史天数
min_history_days = 5

# 触发信号后，建议仓位比例
signal_position_scale = 0

price_figsize = (12, 6)
factor_figsize = (12, 5)
signal_point_size = 20
plot_dpi = 300


# =========================
# 输入路径
# =========================

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

factor_figure_path = (
    f"../../results/figures/{symbol}_{factor_id}_{factor_name}_factor_value.png"
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

if "is_reversal_window" not in daily.columns:
    daily["is_reversal_window"] = 0

if "reversal_segment_id" not in daily.columns:
    daily["reversal_segment_id"] = np.nan


# =========================
# 3. 计算价格上涨比例
# =========================
# return > 0 表示当天价格上涨
# price_up_ratio 表示过去 price_window 天中，上涨天数占比
#
# 注意：
# 这里 price_up_ratio 包含今天的涨跌情况。
# 如果之后真实交易，为了避免用当天收盘后信号交易当天，
# 回测时应该用 signal.shift(1) 后的仓位赚下一天收益。

daily["daily_return"] = daily["close"].pct_change()
daily["is_up_day"] = (daily["daily_return"] > 0).astype(int)

daily["price_up_ratio"] = (
    daily["is_up_day"]
    .rolling(window=price_window, min_periods=min_history_days)
    .mean()
)


# =========================
# 4. 计算持仓量明显下降：MAD 方法
# =========================
# 用今天 log(open_interest) 和过去 oi_window 天比较。
#
# log_open_interest_median_past:
#   过去 oi_window 天 log(open_interest) 中位数，不包含今天
#
# log_open_interest_mad_past:
#   过去 oi_window 天 log(open_interest) 的 MAD，不包含今天
#
# log_open_interest_mad_score:
#   今天持仓量相对过去一段时间的 MAD 标准化偏离

daily.loc[daily["open_interest"] <= 0, "open_interest"] = np.nan
daily["log_open_interest"] = np.log(daily["open_interest"])

daily["delta_log_open_interest"] = (
    daily["log_open_interest"] - daily["log_open_interest"].shift(1)
)

daily["oi_change_rate"] = (
    np.exp(daily["delta_log_open_interest"]) - 1
) * 100

daily["log_open_interest_median_past"] = (
    daily["log_open_interest"]
    .rolling(window=oi_window, min_periods=min_history_days)
    .median()
    .shift(1)
)

daily["log_open_interest_mad_past"] = np.nan

for i in range(len(daily)):

    start_i = max(0, i - oi_window)

    history = daily.iloc[start_i:i]["log_open_interest"].dropna()

    if len(history) < min_history_days:
        continue

    median_value = history.median()

    mad_value = (history - median_value).abs().median()

    daily.loc[daily.index[i], "log_open_interest_mad_past"] = mad_value


daily["log_open_interest_mad_score"] = (
    (
        daily["log_open_interest"]
        - daily["log_open_interest_median_past"]
    )
    / (
        mad_scale * daily["log_open_interest_mad_past"]
        + mad_epsilon
    )
)

# 如果 MAD 为 0，说明过去这段时间持仓量几乎没有波动，
# 此时 MAD score 不稳定，设为 NaN
daily.loc[
    daily["log_open_interest_mad_past"] <= 0,
    "log_open_interest_mad_score"
] = np.nan


# =========================
# 5. 生成因子信号
# =========================
# 这个因子主要用于上涨趋势：
# 价格过去一段时间大部分在涨，同时持仓量明显下降。
#
# 直觉：
# 价格上涨但持仓量下降，可能表示上涨过程伴随减仓、
# 空头回补或多头获利了结，行情可能进入背离/过热阶段。

daily["price_up_oi_down_signal"] = 0

daily.loc[
    (
        daily["price_up_ratio"] >= price_up_ratio_threshold
    ) & (
        daily["log_open_interest_mad_score"] <= oi_mad_threshold
    ) & (
        daily["delta_log_open_interest"] < 0
    ),
    "price_up_oi_down_signal"
] = 1


# =========================
# 6. 生成标准化因子每日表
# =========================

daily["factor_id"] = factor_id
daily["factor_name"] = factor_name

# 主因子值：
# 这里用 log_open_interest_mad_score 作为 factor_value。
# 数值越低，表示持仓量相对过去窗口下降越明显。
daily["factor_value"] = daily["log_open_interest_mad_score"]

daily["signal"] = daily["price_up_oi_down_signal"]

daily["position_scale"] = 1.0
daily.loc[daily["signal"] == 1, "position_scale"] = signal_position_scale

# 是否为有效信号：
# 只有在趋势反转段内触发信号，才算有效
daily["is_effective_signal"] = (
    (daily["signal"] == 1)
    & (daily["is_reversal_window"] == 1)
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

        "daily_return",
        "is_up_day",
        "price_up_ratio",

        "open_interest",
        "log_open_interest",
        "delta_log_open_interest",
        "oi_change_rate",
        "log_open_interest_median_past",
        "log_open_interest_mad_past",
        "log_open_interest_mad_score",

        "signal",
        "position_scale",
        "is_effective_signal",
    ]
].copy()


# =========================
# 7. 生成标准化信号事件表
# =========================

signal_points = daily[daily["signal"] == 1].copy()

print("检测到的信号日期：")
print(signal_points["date"].dt.strftime("%Y-%m-%d").to_list())

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
        "price_up_ratio": row["price_up_ratio"],
        "open_interest": row["open_interest"],
        "log_open_interest": row["log_open_interest"],
        "delta_log_open_interest": row["delta_log_open_interest"],
        "oi_change_rate": row["oi_change_rate"],
        "log_open_interest_mad_score": (
            row["log_open_interest_mad_score"]
        ),

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

        # 整个趋势段内的信号情况
        "signal_days_in_trend": part["signal"].sum(),
        "signal_ratio_in_trend": part["signal"].mean(),

        # 趋势反转段内的信号情况
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

        # 反转段内因子表现
        "min_factor_value_in_reversal_window": (
            reversal_part["factor_value"].min()
        ),
        "mean_factor_value_in_reversal_window": (
            reversal_part["factor_value"].mean()
        ),

        "max_price_up_ratio_in_reversal_window": (
            reversal_part["price_up_ratio"].max()
        ),
        "mean_price_up_ratio_in_reversal_window": (
            reversal_part["price_up_ratio"].mean()
        ),
        "min_log_open_interest_mad_score_in_reversal_window": (
            reversal_part["log_open_interest_mad_score"].min()
        ),
        "mean_log_open_interest_mad_score_in_reversal_window": (
            reversal_part["log_open_interest_mad_score"].mean()
        ),
        "min_oi_change_rate_in_reversal_window": (
            reversal_part["oi_change_rate"].min()
        ),
        "mean_oi_change_rate_in_reversal_window": (
            reversal_part["oi_change_rate"].mean()
        ),
    })

factor_summary = pd.DataFrame(result_rows)


# =========================
# 9. 保存结果表格
# =========================

os.makedirs(os.path.dirname(factor_output_path), exist_ok=True)
os.makedirs(os.path.dirname(signal_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

factor_daily.to_csv(factor_output_path, index=False)
signal_table.to_csv(signal_output_path, index=False)
factor_summary.to_csv(summary_output_path, index=False)

print(f"因子 {factor_id}：价格上涨 + 持仓量明显下降分析完成。")
print(f"因子每日值表保存为：{factor_output_path}")
print(f"因子信号事件表保存为：{signal_output_path}")
print(f"趋势段汇总表保存为：{summary_output_path}")


# =========================
# 10. 画图：价格、信号、有效信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily["close"], label="close")

signal_points = daily[daily["signal"] == 1]
effective_points = daily[daily["is_effective_signal"] == 1]

plt.scatter(
    signal_points["date"],
    signal_points["close"],
    s=signal_point_size,
    label="price up + OI down signal"
)

plt.scatter(
    effective_points["date"],
    effective_points["close"],
    s=signal_point_size * 2,
    marker="X",
    label="effective signal in reversal window"
)

plt.title(f"{symbol} Close Price and Factor 42 Signal")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 11. 画图：价格和持仓量
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


# =========================
# 12. 画图：价格上涨比例和持仓量 MAD score
# =========================

plt.figure(figsize=factor_figsize)

plt.plot(
    daily["date"],
    daily["price_up_ratio"],
    label="price up ratio"
)

plt.plot(
    daily["date"],
    daily["log_open_interest_mad_score"],
    label="OI MAD score"
)

plt.axhline(
    price_up_ratio_threshold,
    linestyle="--",
    label="price up ratio threshold"
)

plt.axhline(
    oi_mad_threshold,
    linestyle=":",
    label="OI MAD threshold"
)

plt.title("Factor 42: Price Up Ratio and OI MAD Score")
plt.xlabel("Date")
plt.ylabel("Factor Value")
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(factor_figure_path), exist_ok=True)
plt.savefig(factor_figure_path, dpi=plot_dpi)
plt.close()

print("\n图片保存为：")
print(price_figure_path)
print(price_oi_figure_path)
print(factor_figure_path)
