from volume_price_factor_utils import (
    load_daily,
    past_rank,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = "LC"

factor_id = "11"
factor_name = "high_speculation"

# history_window 表示“只看今天以前多少个交易日”作为历史参照。
# 这里必须排除今天本身，因为今天的投机度只有在当天数据确认后才知道；
# 如果把今天放进历史样本里，会让分位数天然更接近中间值，也会引入未来函数味道。
history_window = 10

# history_rank_threshold 是触发信号的历史分位数阈值。
# 0.95 的含义是：今天的投机度高于或等于过去 10 个交易日中至少 95% 的观测值；
# 在 history_window=10 时，本质上要求今天投机度处于过去窗口的极高位置。
history_rank_threshold = 0.95

# min_history_days 控制最少需要多少个历史样本才开始计算分位数。
# 当前参数等于 history_window，所以前 10 个交易日不会产生有效分位数；
# 这样可以避免样本太短时，一个偶然的高值就被判成“高投机度”。
min_history_days = 10

# 单因子触发后，建议仓位比例。
# 没有信号时公共输出函数会保持 1.0；有信号时改为这里的 0.5。
signal_position_scale = 0.5


# =========================
# 1. 读取日频数据
# =========================

daily = load_daily(symbol)


# =========================
# 2. 计算投机度历史分位数
# =========================
# past_rank() 做的事情等价于原来的 for 循环：
# 1. 对第 i 天，只取 [i - history_window, i) 这段历史，不包含当天；
# 2. 历史样本数不足 min_history_days 时返回 NaN；
# 3. 分位数 = 历史窗口中“小于等于今天投机度”的样本占比。
# 这样既保留了原来的算法逻辑，也把重复的窗口分位数计算交给公共函数。

daily["speculation_rank_history"] = past_rank(
    daily["speculation"],
    window=history_window,
    min_history_days=min_history_days,
)


# =========================
# 3. 生成高投机度信号
# =========================
# 当历史分位数达到阈值时，说明今天投机度相对过去窗口已经处在极高位置；
# 信号列只负责标记日期，后续仓位缩放由公共输出函数统一处理。

daily["high_speculation_signal"] = (
    daily["speculation_rank_history"] >= history_rank_threshold
).astype(int)


# =========================
# 4. 保存标准化结果
# =========================
# factor_value 仍然使用 speculation_rank_history，和原实现一致。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会同时生成标准字段：factor_id、factor_name、signal、position、position_scale。

feature_columns = [
    "speculation",
    "speculation_rank_history",
]

result = save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="speculation_rank_history",
    signal_column="high_speculation_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "speculation_rank_history",
        "speculation",
    ],
)


# =========================
# 5. 补充因子参数到汇总表
# =========================
# 公共函数已经生成整体汇总表；这里额外写入本因子特有的参数，方便之后
# 只看 summary 文件时也能知道信号来自哪个历史窗口和哪个分位数阈值。

summary_table = result["summary_table"].copy()
summary_table["history_window"] = history_window
summary_table["history_rank_threshold"] = history_rank_threshold
summary_table["min_history_days"] = min_history_days
summary_table["valid_rank_days"] = int(
    daily["speculation_rank_history"].notna().sum()
)
summary_table["max_speculation"] = daily["speculation"].max()
summary_table.to_csv(result["summary_output_path"], index=False)
