from volume_price_factor_utils import (
    load_daily,
    mad_score,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = "LC"

factor_id = "32"
factor_name = "price_down_speculation_up"

# price_window 表示用过去多少个交易日判断“价格大部分时间下跌”。
# 原算法用 daily_return < 0 标记下跌日，再计算最近 price_window 天下跌日占比；
# 这个比例包含今天，因为今天收盘后才能确认今天是否下跌。
price_window = 10

# spec_window 表示投机度 MAD 的历史参照窗口。
# 历史中位数和历史 MAD 都只使用今天以前的数据，不包含今天；
# 今天的投机度只作为待判断值，避免把待判断样本放进自己的基准样本。
spec_window = 10

# 过去 price_window 天中，至少多少比例是下跌日。
# price_down_ratio >= 0.6 表示最近窗口内下跌日占多数，
# 这是原因子里“价格持续走弱”的条件。
price_down_ratio_threshold = 0.6

# 投机度突然升高的阈值。
# MAD score >= 2 表示今天投机度明显高于过去一段时间；
# 最终信号要求价格下跌比例达标且投机度 MAD score 达标。
spec_mad_threshold = 2

# MAD 调整系数和极小值保护。
# 1.4826 把 MAD 调整到类似标准差的尺度；
# 1e-12 防止分母为 0，同时 MAD <= 0 时仍会把 score 设为 NaN。
mad_scale = 1.4826
mad_epsilon = 1e-12

# 最小历史天数。
# 对 price_down_ratio 来说，这是 rolling mean 的 min_periods；
# 对 speculation MAD 来说，这是计算历史中位数和历史 MAD 所需的最少样本数。
min_history_days = 8

# 触发信号后，建议仓位比例。
signal_position_scale = 0


# =========================
# 1. 读取日频数据
# =========================

daily = load_daily(symbol)


# =========================
# 2. 计算价格下跌比例
# =========================
# daily_return < 0 表示当天价格下跌。
# price_down_ratio 表示最近 price_window 天中，下跌天数占比。
# 注意：这里和原实现一样，price_down_ratio 包含今天的涨跌情况；
# 如果之后真实交易或回测，仍应在回测端用 signal.shift(1) 后的仓位赚下一天收益。

daily["daily_return"] = daily["close"].pct_change()
daily["is_down_day"] = (daily["daily_return"] < 0).astype(int)

daily["price_down_ratio"] = (
    daily["is_down_day"]
    .rolling(window=price_window, min_periods=min_history_days)
    .mean()
)


# =========================
# 3. 计算投机度突然升高：MAD 方法
# =========================
# mad_score() 等价于原来的 rolling median + rolling apply(calc_mad)：
# 1. speculation_median_past 是过去 spec_window 天投机度中位数，不包含今天；
# 2. speculation_mad_past 是过去 spec_window 天投机度 MAD，不包含今天；
# 3. speculation_mad_score = (今天投机度 - 历史中位数) / (1.4826 * 历史 MAD + 1e-12)；
# 4. 当历史 MAD <= 0 时，MAD score 设为 NaN，避免无波动窗口导致分数失真。

(
    daily["speculation_median_past"],
    daily["speculation_mad_past"],
    daily["speculation_mad_score"],
) = mad_score(
    daily["speculation"],
    window=spec_window,
    min_history_days=min_history_days,
    mad_scale=mad_scale,
    mad_epsilon=mad_epsilon,
)


# =========================
# 4. 生成因子信号
# =========================
# 这个因子关注价格持续走弱时的投机度升高：
# 价格过去一段时间大部分在跌，同时投机度突然升高。
# 直觉上，价格已经跌了一段但投机度突然升高，可能代表恐慌交易、
# 跟风做空或短线资金集中进入，行情可能进入下跌过热阶段。

daily["price_down_speculation_up_signal"] = (
    (daily["price_down_ratio"] >= price_down_ratio_threshold) &
    (daily["speculation_mad_score"] >= spec_mad_threshold)
).astype(int)


# =========================
# 5. 保存标准化结果
# =========================
# factor_value 仍然使用 speculation_mad_score，和原实现一致。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会统一生成 factor_id、factor_name、signal、position、position_scale，
# 并负责保存 CSV 与基础图形输出。

feature_columns = [
    "daily_return",
    "is_down_day",
    "price_down_ratio",
    "speculation",
    "speculation_median_past",
    "speculation_mad_past",
    "speculation_mad_score",
]

result = save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="speculation_mad_score",
    signal_column="price_down_speculation_up_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "price_down_ratio",
        "speculation_mad_score",
    ],
)


# =========================
# 6. 补充因子参数到汇总表
# =========================
# 公共函数已经生成整体汇总表；这里补上本因子特有参数和原汇总中关注的统计量。
# 这样之后只看 summary 文件，也能知道价格下跌比例窗口、投机度 MAD 窗口、
# 两个触发阈值，以及下跌日比例和 MAD score 的大致分布。

summary_table = result["summary_table"].copy()
summary_table["price_window"] = price_window
summary_table["spec_window"] = spec_window
summary_table["price_down_ratio_threshold"] = price_down_ratio_threshold
summary_table["spec_mad_threshold"] = spec_mad_threshold
summary_table["mad_scale"] = mad_scale
summary_table["mad_epsilon"] = mad_epsilon
summary_table["min_history_days"] = min_history_days
summary_table["down_days"] = int(daily["is_down_day"].sum())
summary_table["down_day_ratio"] = daily["is_down_day"].mean()
summary_table["mean_price_down_ratio"] = daily[
    "price_down_ratio"
].mean()
summary_table["max_price_down_ratio"] = daily[
    "price_down_ratio"
].max()
summary_table["min_price_down_ratio"] = daily[
    "price_down_ratio"
].min()
summary_table["mean_speculation_mad_score"] = daily[
    "speculation_mad_score"
].mean()
summary_table["max_speculation_mad_score"] = daily[
    "speculation_mad_score"
].max()
summary_table["min_speculation_mad_score"] = daily[
    "speculation_mad_score"
].min()
summary_table["mean_daily_return"] = daily["daily_return"].mean()
summary_table["max_daily_return"] = daily["daily_return"].max()
summary_table["min_daily_return"] = daily["daily_return"].min()
summary_table.to_csv(result["summary_output_path"], index=False)
