import numpy as np

from volume_price_factor_utils import (
    load_daily,
    mad_score,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = "LC"

factor_id = "42"
factor_name = "price_up_oi_down"

# price_window 表示用过去多少个交易日判断“价格大部分时间上涨”。
# 原算法用 daily_return > 0 标记上涨日，再计算最近 price_window 天上涨日占比；
# 这个比例包含今天，因为今天收盘后才能确认今天是否上涨。
price_window = 10

# oi_window 表示持仓量历史参照窗口。
# 持仓量先取 log，再和过去 oi_window 天的 log(open_interest) 比较；
# 历史中位数和历史 MAD 都只使用今天以前的数据，不包含今天。
oi_window = 10

# 过去 price_window 天中，至少多少比例是上涨日。
# price_up_ratio >= 0.7 表示最近窗口内上涨日占多数，
# 这是原因子里“价格持续走强”的条件。
price_up_ratio_threshold = 0.7

# 持仓量明显低于过去一段时间的阈值。
# OI MAD score <= -1 表示今天 log(open_interest) 明显低于过去窗口。
oi_mad_threshold = -1

# MAD 缩放系数和极小值保护。
# 1.4826 把 MAD 调整到类似标准差的尺度；
# 1e-12 防止分母为 0，同时 MAD <= 0 时仍会把 score 设为 NaN。
mad_scale = 1.4826
mad_epsilon = 1e-12

# 最小历史天数。
# 对 price_up_ratio 来说，这是 rolling mean 的 min_periods；
# 对 OI MAD 来说，这是计算历史中位数和历史 MAD 所需的最少样本数。
min_history_days = 5

# 触发信号后，建议仓位比例。
signal_position_scale = 0


# =========================
# 1. 读取日频数据
# =========================

daily = load_daily(symbol)


# =========================
# 2. 计算价格上涨比例
# =========================
# daily_return > 0 表示当天价格上涨。
# price_up_ratio 表示最近 price_window 天中，上涨天数占比。
# 注意：这里和原实现一样，price_up_ratio 包含今天的涨跌情况；
# 如果之后真实交易或回测，仍应在回测端用 signal.shift(1) 后的仓位赚下一天收益。

daily["daily_return"] = daily["close"].pct_change()
daily["is_up_day"] = (daily["daily_return"] > 0).astype(int)

daily["price_up_ratio"] = (
    daily["is_up_day"]
    .rolling(window=price_window, min_periods=min_history_days)
    .mean()
)


# =========================
# 3. 计算持仓量明显下降：MAD 方法
# =========================
# 原算法先把 open_interest <= 0 的值设为 NaN，再对 open_interest 取 log。
# 这样可以避免对非正数取对数，也让后续历史窗口自动忽略无效持仓量。
# delta_log_open_interest < 0 是最终信号的额外条件，
# 它确保今天持仓量相对上一交易日确实下降，而不是只在历史窗口中偏低。

daily.loc[daily["open_interest"] <= 0, "open_interest"] = np.nan
daily["log_open_interest"] = np.log(daily["open_interest"])

daily["delta_log_open_interest"] = (
    daily["log_open_interest"] - daily["log_open_interest"].shift(1)
)

daily["oi_change_rate"] = (
    np.exp(daily["delta_log_open_interest"]) - 1
) * 100

# mad_score() 等价于原来的手写循环：
# 1. log_open_interest_median_past 是过去 oi_window 天 log(OI) 中位数，不包含今天；
# 2. log_open_interest_mad_past 是过去 oi_window 天 log(OI) 的 MAD，不包含今天；
# 3. log_open_interest_mad_score = (今天 log(OI) - 历史中位数) / (1.4826 * 历史 MAD + 1e-12)；
# 4. 当历史 MAD <= 0 时，MAD score 设为 NaN，避免无波动窗口导致分数失真。
(
    daily["log_open_interest_median_past"],
    daily["log_open_interest_mad_past"],
    daily["log_open_interest_mad_score"],
) = mad_score(
    daily["log_open_interest"],
    window=oi_window,
    min_history_days=min_history_days,
    mad_scale=mad_scale,
    mad_epsilon=mad_epsilon,
)


# =========================
# 4. 生成因子信号
# =========================
# 这个因子关注价格持续走强时的持仓量明显下降：
# 价格过去一段时间大部分在涨，同时持仓量相对历史窗口明显偏低，
# 且今天持仓量确实比上一交易日下降。
# 直觉上，价格上涨但持仓量下降，可能表示上涨过程伴随减仓、
# 空头回补或多头获利了结，行情可能进入背离/过热阶段。

daily["price_up_oi_down_signal"] = (
    (daily["price_up_ratio"] >= price_up_ratio_threshold) &
    (daily["log_open_interest_mad_score"] <= oi_mad_threshold) &
    (daily["delta_log_open_interest"] < 0)
).astype(int)


# =========================
# 5. 保存标准化结果
# =========================
# factor_value 仍然使用 log_open_interest_mad_score，和原实现一致。
# 数值越低，表示持仓量相对过去窗口下降越明显。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会统一生成 factor_id、factor_name、signal、position、position_scale，
# 并负责保存 CSV 与基础图形输出。

feature_columns = [
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
]

result = save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="log_open_interest_mad_score",
    signal_column="price_up_oi_down_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "price_up_ratio",
        "log_open_interest_mad_score",
    ],
)


# =========================
# 6. 补充因子参数到汇总表
# =========================
# 公共函数已经生成整体汇总表；这里补上本因子特有参数和原汇总中关注的统计量。
# 这样之后只看 summary 文件，也能知道价格上涨比例窗口、OI MAD 窗口、
# 两个触发阈值，以及 OI 变化率和 MAD score 的大致分布。

summary_table = result["summary_table"].copy()
summary_table["price_window"] = price_window
summary_table["oi_window"] = oi_window
summary_table["price_up_ratio_threshold"] = price_up_ratio_threshold
summary_table["oi_mad_threshold"] = oi_mad_threshold
summary_table["mad_scale"] = mad_scale
summary_table["mad_epsilon"] = mad_epsilon
summary_table["min_history_days"] = min_history_days
summary_table["up_days"] = int(daily["is_up_day"].sum())
summary_table["up_day_ratio"] = daily["is_up_day"].mean()
summary_table["mean_price_up_ratio"] = daily["price_up_ratio"].mean()
summary_table["max_price_up_ratio"] = daily["price_up_ratio"].max()
summary_table["min_price_up_ratio"] = daily["price_up_ratio"].min()
summary_table["mean_log_open_interest_mad_score"] = daily[
    "log_open_interest_mad_score"
].mean()
summary_table["max_log_open_interest_mad_score"] = daily[
    "log_open_interest_mad_score"
].max()
summary_table["min_log_open_interest_mad_score"] = daily[
    "log_open_interest_mad_score"
].min()
summary_table["mean_oi_change_rate"] = daily["oi_change_rate"].mean()
summary_table["max_oi_change_rate"] = daily["oi_change_rate"].max()
summary_table["min_oi_change_rate"] = daily["oi_change_rate"].min()
summary_table["mean_daily_return"] = daily["daily_return"].mean()
summary_table["max_daily_return"] = daily["daily_return"].max()
summary_table["min_daily_return"] = daily["daily_return"].min()
summary_table.to_csv(result["summary_output_path"], index=False)
