from volume_price_factor_utils import (
    load_daily,
    past_rank,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = "LC"

factor_id = "22"
factor_name = "speculation_first_difference"

# diff_days 表示和多少个交易日前相比，计算投机度一阶差分。
# 当前为 1，所以 DeltaSpec_t = Spec_t - Spec_{t-1}；
# DeltaSpec_t > 0 表示投机度上升，DeltaSpec_t < 0 表示投机度回落。
diff_days = 1

# history_window 表示用过去多少个交易日判断一阶差分是否异常。
# 分位数计算只使用今天以前的数据，不包含今天本身，避免把待判断样本放进参照集。
history_window = 20

# high_diff_rank_threshold 和 low_diff_rank_threshold 用来标记方向性异常：
# 一阶差分处于历史高位，说明投机度突然大幅上升；
# 一阶差分处于历史低位，说明投机度突然大幅下降。
high_diff_rank_threshold = 0.98
low_diff_rank_threshold = 0.02

# large_abs_diff_rank_threshold 用来标记“幅度异常”。
# 这个阈值不关心方向，只看 abs(DeltaSpec) 是否处在过去窗口的高位；
# 原实现最终 signal 使用的就是这个绝对一阶差分信号。
large_abs_diff_rank_threshold = 0.95

# 计算历史分位数所需的最少历史天数。
# 有效历史样本不足时，rank 保持 NaN，不产生方向性信号或最终信号。
min_history_days = 10

# 单因子触发后，建议仓位比例。
signal_position_scale = 0


# =========================
# 1. 读取日频数据
# =========================

daily = load_daily(symbol)


# =========================
# 2. 计算投机度一阶差分
# =========================
# speculation_first_diff 是投机度的方向性变化：
#   Spec_t - Spec_{t-diff_days}
# speculation_abs_first_diff 是变化幅度：
#   abs(speculation_first_diff)
# 二者都保留，因为前者用于判断上升/下降方向，后者用于最终的大幅变化信号。

daily["speculation_first_diff"] = (
    daily["speculation"] - daily["speculation"].shift(diff_days)
)

daily["speculation_abs_first_diff"] = (
    daily["speculation_first_diff"].abs()
)


# =========================
# 3. 判断一阶差分是否处于历史异常位置
# =========================
# past_rank() 等价于原来的逐行循环：
# 1. 对第 i 天，只取 [i - history_window, i) 的历史数据；
# 2. 历史窗口会 dropna，当前值为 NaN 时也返回 NaN；
# 3. 有效历史样本不足 min_history_days 时返回 NaN；
# 4. 分位数 = 历史窗口中“小于等于今天值”的样本占比。

daily["speculation_first_diff_rank"] = past_rank(
    daily["speculation_first_diff"],
    window=history_window,
    min_history_days=min_history_days,
)

daily["speculation_abs_first_diff_rank"] = past_rank(
    daily["speculation_abs_first_diff"],
    window=history_window,
    min_history_days=min_history_days,
)


# =========================
# 4. 生成一阶差分异常信号
# =========================
# 方向性信号：
# - rise_signal：一阶差分处于历史高位，表示投机度短期显著上升；
# - drop_signal：一阶差分处于历史低位，表示投机度短期显著下降。
# 最终信号：
# - large_abs_signal：绝对一阶差分处于历史高位，表示投机度发生异常剧烈变化。
# 注意：最终 factor signal 仍然只使用 large_abs_signal，和原实现一致。

daily["speculation_first_diff_rise_signal"] = (
    daily["speculation_first_diff_rank"] >= high_diff_rank_threshold
).astype(int)

daily["speculation_first_diff_drop_signal"] = (
    daily["speculation_first_diff_rank"] <= low_diff_rank_threshold
).astype(int)

daily["speculation_first_diff_large_abs_signal"] = (
    daily["speculation_abs_first_diff_rank"] >= (
        large_abs_diff_rank_threshold
    )
).astype(int)


# =========================
# 5. 保存标准化结果
# =========================
# factor_value 仍然使用 speculation_abs_first_diff_rank，和原实现一致。
# 这表示：投机度变化幅度是否相对历史窗口处在异常高位。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会统一生成 factor_id、factor_name、signal、position、position_scale，
# 并负责保存 CSV 与基础图形输出。

feature_columns = [
    "speculation",
    "speculation_first_diff",
    "speculation_abs_first_diff",
    "speculation_first_diff_rank",
    "speculation_abs_first_diff_rank",
    "speculation_first_diff_rise_signal",
    "speculation_first_diff_drop_signal",
    "speculation_first_diff_large_abs_signal",
]

result = save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="speculation_abs_first_diff_rank",
    signal_column="speculation_first_diff_large_abs_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "speculation_first_diff_rank",
        "speculation_abs_first_diff_rank",
    ],
)


# =========================
# 6. 补充因子参数到汇总表
# =========================
# 公共函数已经生成整体汇总表；这里补上本因子特有参数和原汇总中关注的统计量。
# 这样之后只看 summary 文件，也能知道差分间隔、历史窗口、上下分位阈值、
# 绝对差分阈值，以及方向性信号和最终大幅变化信号分别触发了多少天。

summary_table = result["summary_table"].copy()
summary_table["diff_days"] = diff_days
summary_table["history_window"] = history_window
summary_table["high_diff_rank_threshold"] = high_diff_rank_threshold
summary_table["low_diff_rank_threshold"] = low_diff_rank_threshold
summary_table["large_abs_diff_rank_threshold"] = (
    large_abs_diff_rank_threshold
)
summary_table["min_history_days"] = min_history_days
summary_table["valid_rank_days"] = int(
    daily["speculation_first_diff_rank"].notna().sum()
)
summary_table["valid_abs_rank_days"] = int(
    daily["speculation_abs_first_diff_rank"].notna().sum()
)
summary_table["rise_signal_days"] = int(
    daily["speculation_first_diff_rise_signal"].sum()
)
summary_table["drop_signal_days"] = int(
    daily["speculation_first_diff_drop_signal"].sum()
)
summary_table["large_abs_signal_days"] = int(
    daily["speculation_first_diff_large_abs_signal"].sum()
)
summary_table["mean_first_diff"] = daily[
    "speculation_first_diff"
].mean()
summary_table["max_first_diff"] = daily[
    "speculation_first_diff"
].max()
summary_table["min_first_diff"] = daily[
    "speculation_first_diff"
].min()
summary_table["mean_abs_first_diff"] = daily[
    "speculation_abs_first_diff"
].mean()
summary_table["max_abs_first_diff"] = daily[
    "speculation_abs_first_diff"
].max()
summary_table.to_csv(result["summary_output_path"], index=False)
