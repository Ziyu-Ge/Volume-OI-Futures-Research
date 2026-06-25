import numpy as np

from volume_price_factor_utils import (
    SYMBOL,
    load_daily,
    mad_score,
    parse_factor_script_metadata,
    past_rank,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = SYMBOL

factor_id, factor_name = parse_factor_script_metadata(__file__)

# 价格均线多头排列窗口。
# ma5 > ma10 > ma20 表示短期价格均线高于中期，中期高于长期。
ma_short_window = 5
ma_mid_window = 30
ma_long_window = 60

# oi_window 表示持仓量历史参照窗口。
# 持仓量先取 log，再和过去 oi_window 天的 log(open_interest) 比较；
# 历史中位数和历史 MAD 都只使用今天以前的数据，不包含今天。
oi_window = 10

# oi_rank_window 表示用过去多少个交易日判断持仓量高位。
# open_interest_rank_10 >= 0.8 表示今天持仓量处在过去 10 日的 80% 高位。
oi_rank_window = 10

# 持仓量高位阈值。
oi_rank_threshold = 0.8

# MAD 缩放系数和极小值保护。
# 1.4826 把 MAD 调整到类似标准差的尺度；
# 1e-12 防止分母为 0，同时 MAD <= 0 时仍会把 score 设为 NaN。
mad_scale = 1.4826
mad_epsilon = 1e-12

# 最小历史天数。
# 对 OI rank 和 OI MAD 来说，这是计算历史参照所需的最少样本数。
min_history_days = 5

# 触发信号后的次一交易日，根据开盘价和前一交易日均值决定方向。
# 这里的均值使用日线 OHLC4 均价，避免依赖不同品种成交额乘数。
# 公共 position 字段保留为 1.0，真实回测方向写入 trade_open_mean_position。
signal_position_scale = 1


# =========================
# 1. 读取日频数据
# =========================

daily = load_daily(symbol)


# =========================
# 2. 计算价格均线多头排列
# =========================
# price_ma_5 > price_ma_10 > price_ma_20 表示价格均线呈多头排列。
# 注意：价格均线包含今天的收盘价，因为今天收盘后才能确认今天的均线状态；
# 如果之后真实交易或回测，仍应在回测端用 signal.shift(1) 后的仓位赚下一天收益。

daily["daily_return"] = daily["close"].pct_change()

daily["price_ma_5"] = (
    daily["close"]
    .rolling(window=ma_short_window, min_periods=ma_short_window)
    .mean()
)
daily["price_ma_10"] = (
    daily["close"]
    .rolling(window=ma_mid_window, min_periods=ma_mid_window)
    .mean()
)
daily["price_ma_20"] = (
    daily["close"]
    .rolling(window=ma_long_window, min_periods=ma_long_window)
    .mean()
)

daily["is_ma_bullish"] = (
    (daily["price_ma_5"] > daily["price_ma_10"]) &
    (daily["price_ma_10"] > daily["price_ma_20"])
).astype(int)


# =========================
# 3. 计算持仓量高位和持仓量变化
# =========================
# 原算法先把 open_interest <= 0 的值设为 NaN，再对 open_interest 取 log。
# 这样可以避免对非正数取对数，也让后续历史窗口自动忽略无效持仓量。
# open_interest_rank_10 使用今天以前的 10 日历史样本比较当前持仓量，
# 用来判断今天持仓量是否处在近期高位。
# delta_log_open_interest < 0 是最终信号的额外条件，
# 它确保今天持仓量相对上一交易日确实下降。

daily.loc[daily["open_interest"] <= 0, "open_interest"] = np.nan
daily["open_interest_rank_10"] = past_rank(
    daily["open_interest"],
    window=oi_rank_window,
    min_history_days=min_history_days,
)
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
# 这个因子关注价格持续走强时的持仓量从高位回落：
# 价格 ma5 > ma10 > ma20，同时持仓量处在 10 日 80% 高位，
# 且今天持仓量确实比上一交易日下降。
# 直觉上，价格上涨但高位持仓开始下降，可能表示上涨过程伴随减仓、
# 空头回补或多头获利了结，行情可能进入背离/过热阶段。

daily["price_up_oi_down_signal"] = (
    (daily["is_ma_bullish"] == 1) &
    (daily["open_interest_rank_10"] >= oi_rank_threshold) &
    (daily["delta_log_open_interest"] < 0)
).astype(int)


# =========================
# 5. 生成信号次日开盘方向
# =========================
# 信号在当天收盘后确认，因此只从次一交易日开始交易。
# 若次日开盘价高于信号日 OHLC4 均价，次日做多；
# 若次日开盘价低于信号日 OHLC4 均价，次日做空；
# 若价格刚好相等或缺少均值，则次日空仓。

daily["day_mean_price"] = (
    daily[["open", "high", "low", "close"]]
    .mean(axis=1)
)

daily["next_day_open"] = daily["open"].shift(-1)
daily["next_day_open_vs_day_mean"] = (
    daily["next_day_open"] - daily["day_mean_price"]
)
daily["signal_next_day_position"] = np.nan

signal_mask = daily["price_up_oi_down_signal"] == 1
next_day_long_mask = (
    signal_mask &
    (daily["next_day_open"] > daily["day_mean_price"])
)
next_day_short_mask = (
    signal_mask &
    (daily["next_day_open"] < daily["day_mean_price"])
)
next_day_flat_mask = (
    signal_mask &
    ~(next_day_long_mask | next_day_short_mask)
)

daily.loc[next_day_long_mask, "signal_next_day_position"] = 1.0
daily.loc[next_day_short_mask, "signal_next_day_position"] = -1.0
daily.loc[next_day_flat_mask, "signal_next_day_position"] = 0.0

daily["previous_day_mean_price"] = daily["day_mean_price"].shift(1)
daily["open_vs_previous_day_mean"] = (
    daily["open"] - daily["previous_day_mean_price"]
)
daily["trade_from_previous_signal"] = (
    daily["price_up_oi_down_signal"]
    .shift(1)
    .fillna(0)
    .astype(int)
)

daily["trade_open_mean_position"] = 1.0
trade_signal_mask = daily["trade_from_previous_signal"] == 1
trade_long_mask = (
    trade_signal_mask &
    (daily["open"] > daily["previous_day_mean_price"])
)
trade_short_mask = (
    trade_signal_mask &
    (daily["open"] < daily["previous_day_mean_price"])
)
trade_flat_mask = (
    trade_signal_mask &
    ~(trade_long_mask | trade_short_mask)
)

daily.loc[trade_long_mask, "trade_open_mean_position"] = 1.0
daily.loc[trade_short_mask, "trade_open_mean_position"] = -1.0
daily.loc[trade_flat_mask, "trade_open_mean_position"] = 0.0


# =========================
# 6. 保存标准化结果
# =========================
# factor_value 仍然使用 log_open_interest_mad_score，和原实现一致；
# signal 中的持仓量条件改为 open_interest_rank_10 >= 0.8。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会统一生成 factor_id、factor_name、signal、position、position_scale。
# backtest_sum.py 会优先使用 trade_open_mean_position 作为交易当天仓位。

feature_columns = [
    "daily_return",
    "price_ma_5",
    "price_ma_10",
    "price_ma_20",
    "is_ma_bullish",
    "open_interest",
    "open_interest_rank_10",
    "log_open_interest",
    "delta_log_open_interest",
    "oi_change_rate",
    "log_open_interest_median_past",
    "log_open_interest_mad_past",
    "log_open_interest_mad_score",
    "day_mean_price",
    "next_day_open",
    "next_day_open_vs_day_mean",
    "signal_next_day_position",
    "previous_day_mean_price",
    "open_vs_previous_day_mean",
    "trade_from_previous_signal",
    "trade_open_mean_position",
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
        "price_ma_5",
        "price_ma_10",
        "price_ma_20",
        "open_interest_rank_10",
        "log_open_interest_mad_score",
    ],
)


# =========================
# 7. 补充因子参数到汇总表
# =========================
# 公共函数已经生成整体汇总表；这里补上本因子特有参数和原汇总中关注的统计量。
# 这样之后只看 summary 文件，也能知道价格均线窗口、OI rank 窗口、
# 触发阈值，以及 OI 变化率、rank 和 MAD score 的大致分布。

summary_table = result["summary_table"].copy()
summary_table["ma_short_window"] = ma_short_window
summary_table["ma_mid_window"] = ma_mid_window
summary_table["ma_long_window"] = ma_long_window
summary_table["oi_window"] = oi_window
summary_table["oi_rank_window"] = oi_rank_window
summary_table["oi_rank_threshold"] = oi_rank_threshold
summary_table["mad_scale"] = mad_scale
summary_table["mad_epsilon"] = mad_epsilon
summary_table["min_history_days"] = min_history_days
summary_table["ma_bullish_days"] = int(daily["is_ma_bullish"].sum())
summary_table["ma_bullish_ratio"] = daily["is_ma_bullish"].mean()
summary_table["mean_open_interest_rank_10"] = daily[
    "open_interest_rank_10"
].mean()
summary_table["max_open_interest_rank_10"] = daily[
    "open_interest_rank_10"
].max()
summary_table["min_open_interest_rank_10"] = daily[
    "open_interest_rank_10"
].min()
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
