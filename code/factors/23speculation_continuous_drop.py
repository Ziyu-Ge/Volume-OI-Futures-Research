import numpy as np
import pandas as pd

from volume_price_factor_utils import (
    load_daily,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = "LC"

factor_id = "23"
factor_name = "speculation_continuous_drop"

# diff_days 表示和多少个交易日前相比，计算投机度一阶差分。
# 当前为 1，所以 DeltaSpec_t = Spec_t - Spec_{t-1}；
# DeltaSpec_t < 0 表示投机度比上一交易日下降。
diff_days = 1

# down_window 表示滚动统计下降天数时回看多少个交易日。
# 原算法统计的是最近 down_window 天中，有多少天的一阶差分满足“下降”条件。
down_window = 5

# down_days_threshold 是最终信号阈值。
# 当最近 down_window 天内投机度下降天数达到该值时，认为投机度出现连续回落压力。
down_days_threshold = 4

# rolling(..., min_periods=min_down_window_days) 的最少有效样本要求。
# 当前为 3，表示早期窗口只要有至少 3 个非 NaN 的 down_flag，就开始输出下降天数；
# 这会影响样本最前面的几天，不能随意改成 down_window。
min_down_window_days = 3

# strictly_negative_diff 控制“下降”的定义：
# True 时只有 DeltaSpec < 0 才算下降；
# False 时 DeltaSpec <= 0 也算下降。
# 当前保持 True，和原实现一致。
strictly_negative_diff = True

# 单因子触发后，建议仓位比例。
signal_position_scale = 0


# =========================
# 1. 读取日频数据
# =========================

daily = load_daily(symbol)


# =========================
# 2. 计算投机度连续回落因子
# =========================
# DeltaSpec_t = Spec_t - Spec_{t-diff_days}
# speculation_down_flag 是单日下降标记：
# - 1.0 表示该日投机度下降；
# - 0.0 表示该日投机度没有下降；
# - NaN 表示无法判断，例如第一天没有上一期数据。
# 后续 rolling sum 会自动跳过 NaN，但仍受 min_down_window_days 约束。

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
    "speculation_down_flag",
] = np.nan

# speculation_down_days 是滚动窗口内的下降天数。
# 这不是“连续 streak”，而是最近 down_window 天里下降标记的数量；
# 原因子最终 signal 用的就是这个 rolling sum。
daily["speculation_down_days"] = (
    daily["speculation_down_flag"]
    .rolling(window=down_window, min_periods=min_down_window_days)
    .sum()
)


# =========================
# 3. 计算连续下降 streak
# =========================
# speculation_down_streak 记录“截至今天已经连续下降了多少天”：
# - 遇到 NaN，说明无法判断下降，streak 重置为 0；
# - down_flag == 1 时，streak 加 1；
# - down_flag == 0 时，streak 重置为 0。
# 这个字段不参与最终 signal，但会进入输出表和汇总表，方便区分
# “窗口内多数天下降”和“真正连续每天下降”这两种不同状态。

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


# =========================
# 4. 生成连续回落信号
# =========================
# 最终信号只看 rolling 下降天数：
# speculation_down_days >= down_days_threshold。
# 注意这里保留 >=，不改成 >；否则刚好等于阈值的日期会被漏掉。

daily["speculation_continuous_drop_signal"] = (
    daily["speculation_down_days"] >= down_days_threshold
).astype(int)


# =========================
# 5. 保存标准化结果
# =========================
# factor_value 仍然使用 speculation_down_days，和原实现一致。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会统一生成 factor_id、factor_name、signal、position、position_scale，
# 并负责保存 CSV 与基础图形输出。

feature_columns = [
    "speculation",
    "speculation_first_diff",
    "speculation_down_flag",
    "speculation_down_days",
    "speculation_down_streak",
]

result = save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="speculation_down_days",
    signal_column="speculation_continuous_drop_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "speculation_down_days",
        "speculation_down_streak",
    ],
)


# =========================
# 6. 补充因子参数到汇总表
# =========================
# 公共函数已经生成整体汇总表；这里补上本因子特有参数和原汇总中关注的统计量。
# 这样之后只看 summary 文件，也能知道差分间隔、滚动窗口长度、触发阈值、
# 最少有效窗口天数、下降定义，以及下降 streak 和一阶差分的大致分布。

summary_table = result["summary_table"].copy()
summary_table["diff_days"] = diff_days
summary_table["down_window"] = down_window
summary_table["down_days_threshold"] = down_days_threshold
summary_table["min_down_window_days"] = min_down_window_days
summary_table["strictly_negative_diff"] = strictly_negative_diff
summary_table["valid_down_days"] = int(
    daily["speculation_down_days"].notna().sum()
)
summary_table["mean_down_streak"] = daily[
    "speculation_down_streak"
].mean()
summary_table["max_down_streak"] = daily[
    "speculation_down_streak"
].max()
summary_table["mean_first_diff"] = daily[
    "speculation_first_diff"
].mean()
summary_table["max_first_diff"] = daily[
    "speculation_first_diff"
].max()
summary_table["min_first_diff"] = daily[
    "speculation_first_diff"
].min()
summary_table["down_flag_days"] = daily[
    "speculation_down_flag"
].fillna(0).sum()
summary_table.to_csv(result["summary_output_path"], index=False)
