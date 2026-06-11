import numpy as np

from volume_price_factor_utils import (
    load_daily,
    past_rank,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = "LC"

factor_id = "21"
factor_name = "speculation_change_rate"

# change_days 表示投机度变化量的比较间隔。
# 当前为 1，所以今天的变化量 = 今天 speculation - 昨天 speculation。
# 后面再用 exp(change) - 1 转成相对变化率；这个写法沿用原实现，
# 不改成 pct_change，因为 speculation 本身不是普通价格，原算法用的是对数差转比例。
change_days = 1

# history_window 表示判断“变化率是否异常”时回看多少个历史交易日。
# 分位数只使用今天以前的数据，不包含今天本身，避免把待判断样本放进参照集。
history_window = 20

# 高低分位阈值分别用于捕捉投机度变化率的极端上升和极端下降。
# 0.99 表示处在历史窗口极高位置，0.01 表示处在历史窗口极低位置；
# 两边任意一边触发，都会形成最终的异常变化率信号。
high_change_rank_threshold = 0.99
low_change_rank_threshold = 0.01

# 最少历史样本数。当前设置为 10，因此即使 history_window 是 20，
# 在有效历史变化率少于 10 个时也不会计算分位数，避免早期样本太少导致误判。
min_history_days = 10

# 单因子触发后，建议仓位比例。
signal_position_scale = 0


# =========================
# 1. 读取日频数据
# =========================

daily = load_daily(symbol)


# =========================
# 2. 计算投机度变化率
# =========================
# 原算法分两步：
# 1. speculation_change 是当前投机度和 change_days 天前投机度的差；
# 2. speculation_change_rate = exp(speculation_change) - 1。
# 这里保持这个转换方式不变，因为它等价于把“对数式变化”还原成比例变化，
# 和简单差分或普通百分比变化不是同一个东西。

daily["speculation_change"] = (
    daily["speculation"] - daily["speculation"].shift(change_days)
)
daily["speculation_change_rate"] = (
    np.exp(daily["speculation_change"]) - 1
)


# =========================
# 3. 判断变化率是否处于历史高位或低位
# =========================
# past_rank() 等价于原来的逐行循环：
# 1. 对第 i 天，只取 [i - history_window, i) 的历史变化率；
# 2. 历史窗口会 dropna，当前变化率为 NaN 时也返回 NaN；
# 3. 有效历史样本不足 min_history_days 时返回 NaN；
# 4. 分位数 = 历史窗口中“小于等于今天变化率”的样本占比。

daily["speculation_change_rate_rank"] = past_rank(
    daily["speculation_change_rate"],
    window=history_window,
    min_history_days=min_history_days,
)


# =========================
# 4. 生成投机度变化率异常信号
# =========================
# fast_rise 捕捉变化率历史分位数达到高阈值的日期；
# fast_drop 捕捉变化率历史分位数达到低阈值的日期。
# 最终信号是二者取或，signal_type 只是用于输出时区分信号来源，不参与计算。

daily["speculation_fast_rise_signal"] = (
    daily["speculation_change_rate_rank"] >= high_change_rank_threshold
).astype(int)

daily["speculation_fast_drop_signal"] = (
    daily["speculation_change_rate_rank"] <= low_change_rank_threshold
).astype(int)

daily["speculation_change_rate_signal"] = (
    (daily["speculation_fast_rise_signal"] == 1)
    | (daily["speculation_fast_drop_signal"] == 1)
).astype(int)

daily["signal_type"] = "none"
daily.loc[
    daily["speculation_fast_rise_signal"] == 1,
    "signal_type",
] = "fast_rise"
daily.loc[
    daily["speculation_fast_drop_signal"] == 1,
    "signal_type",
] = "fast_drop"
daily.loc[
    (
        daily["speculation_fast_rise_signal"] == 1
    ) & (
        daily["speculation_fast_drop_signal"] == 1
    ),
    "signal_type",
] = "both"


# =========================
# 5. 保存标准化结果
# =========================
# factor_value 仍然使用 speculation_change_rate，和原实现一致。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会统一生成 factor_id、factor_name、signal、position、position_scale，
# 并负责保存 CSV 与基础图形输出。

feature_columns = [
    "speculation",
    "speculation_change",
    "speculation_change_rate",
    "speculation_change_rate_rank",
    "speculation_fast_rise_signal",
    "speculation_fast_drop_signal",
    "signal_type",
]

result = save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="speculation_change_rate",
    signal_column="speculation_change_rate_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "speculation_change_rate",
        "speculation_change_rate_rank",
    ],
)


# =========================
# 6. 补充因子参数到汇总表
# =========================
# 公共函数已经生成整体汇总表；这里补上本因子特有参数和原汇总中关注的统计量。
# 这样只看 summary 文件时，也能知道变化率间隔、历史窗口、上下分位阈值，
# 以及异常上升和异常下降分别触发了多少天。

summary_table = result["summary_table"].copy()
summary_table["change_days"] = change_days
summary_table["history_window"] = history_window
summary_table["high_change_rank_threshold"] = high_change_rank_threshold
summary_table["low_change_rank_threshold"] = low_change_rank_threshold
summary_table["min_history_days"] = min_history_days
summary_table["valid_rank_days"] = int(
    daily["speculation_change_rate_rank"].notna().sum()
)
summary_table["fast_rise_days"] = int(
    daily["speculation_fast_rise_signal"].sum()
)
summary_table["fast_drop_days"] = int(
    daily["speculation_fast_drop_signal"].sum()
)
summary_table["mean_rank"] = daily["speculation_change_rate_rank"].mean()
summary_table["max_rank"] = daily["speculation_change_rate_rank"].max()
summary_table["min_rank"] = daily["speculation_change_rate_rank"].min()
summary_table.to_csv(result["summary_output_path"], index=False)
