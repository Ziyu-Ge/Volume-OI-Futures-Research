from volume_price_factor_utils import (
    load_daily,
    past_mad_score,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = "LC"

factor_id = "12"
factor_name = "speculation_mad"

# history_window 表示每一天只和“今天以前”的最近多少个交易日比较。
# 这里不把今天放入历史窗口：今天的投机度是待判断对象，历史中位数和 MAD
# 必须完全由昨天及以前的数据给出，才能避免未来数据泄露。
history_window = 10

# min_history_days 是开始计算 MAD 标准化偏离前要求的最少历史样本数。
# 当前值等于 history_window，因此前 10 个交易日不会产生有效 MAD 值；
# 这样可以避免样本太少时，中位数和 MAD 被个别观测值强烈影响。
min_history_days = 10

# mad_threshold 是信号触发阈值。
# 因子值使用 abs(speculation_mad_value) > mad_threshold 判断，
# 所以投机度相对历史窗口“异常偏高”和“异常偏低”都会触发信号。
mad_threshold = 3

# mad_scale=1.4826 是把 MAD 调整到类似标准差尺度的常用缩放系数。
# mad_epsilon 只用于判断历史 MAD 是否近似为 0；只要 MAD 不接近 0，
# 原算法就直接除以 mad_scale * MAD，不额外把 epsilon 加进分母。
mad_scale = 1.4826
mad_epsilon = 1e-12

# 单因子触发后，建议仓位比例。
signal_position_scale = 0


# =========================
# 1. 读取日频数据
# =========================

daily = load_daily(symbol)


# =========================
# 2. 计算投机度 MAD 稳健偏离因子
# =========================
# past_mad_score() 精确复刻原来的逐行循环：
# 1. 对第 i 天，只取 [i - history_window, i) 这段历史，不包含当天；
# 2. 历史样本数不足 min_history_days，或当天投机度缺失时，返回 NaN；
# 3. 历史中位数 = 历史窗口内 speculation 的 median；
# 4. 历史 MAD = abs(speculation - 历史中位数) 的 median；
# 5. 如果历史 MAD 接近 0，则跳过，避免除以一个没有稳定意义的极小数；
# 6. MAD 标准化偏离 = (今天投机度 - 历史中位数) / (mad_scale * 历史 MAD)。

(
    daily["speculation_median_history"],
    daily["speculation_mad_history"],
    daily["speculation_mad_value"],
) = past_mad_score(
    daily["speculation"],
    window=history_window,
    min_history_days=min_history_days,
    mad_scale=mad_scale,
    mad_epsilon=mad_epsilon,
)


# =========================
# 3. 生成 MAD 极端偏离信号
# =========================
# 这里沿用原来的严格大于号：abs(MAD value) > 3 才触发信号。
# 如果刚好等于 3，不触发；这个边界条件不要改成 >=，否则信号日期可能变化。

daily["speculation_mad_signal"] = (
    daily["speculation_mad_value"].abs() > mad_threshold
).astype(int)


# =========================
# 4. 保存标准化结果
# =========================
# factor_value 仍然使用 speculation_mad_value，和原实现一致。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会统一生成 factor_id、factor_name、signal、position、position_scale，
# 并负责保存 CSV 与基础图形输出。

feature_columns = [
    "speculation",
    "speculation_median_history",
    "speculation_mad_history",
    "speculation_mad_value",
]

result = save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="speculation_mad_value",
    signal_column="speculation_mad_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "speculation_mad_value",
    ],
)


# =========================
# 5. 补充因子参数到汇总表
# =========================
# 公共函数已经生成整体汇总表；这里补上本因子特有参数和统计量。
# 这样之后只打开 summary 文件，也能看出当前结果使用了多长历史窗口、
# 最少样本要求、MAD 阈值，以及投机度原始序列的大致范围。

summary_table = result["summary_table"].copy()
summary_table["history_window"] = history_window
summary_table["min_history_days"] = min_history_days
summary_table["mad_threshold"] = mad_threshold
summary_table["mad_scale"] = mad_scale
summary_table["mad_epsilon"] = mad_epsilon
summary_table["valid_mad_days"] = int(
    daily["speculation_mad_value"].notna().sum()
)
summary_table["max_speculation"] = daily["speculation"].max()
summary_table["min_speculation"] = daily["speculation"].min()
summary_table.to_csv(result["summary_output_path"], index=False)
