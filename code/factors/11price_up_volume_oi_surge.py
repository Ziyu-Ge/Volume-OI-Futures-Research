import numpy as np

from volume_price_factor_utils import (
    SYMBOL,
    load_daily,
    mad_score,
    parse_factor_script_metadata,
    past_rank,
    positive_part,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = SYMBOL

factor_id, factor_name = parse_factor_script_metadata(__file__)

# 价格、持仓变化异常和成交量异常的触发阈值。
# close_rank_20 >= 0.70 表示价格处在过去 20 日的偏高位置；
# oi_change_5_rate_mad_abs_score >= 1.0 表示 5 日持仓变化率明显偏离常态；
# volume_mad_score >= 1.0 表示成交量相对过去窗口明显放大。
price_rank_threshold = 0.90
oi_change_mad_threshold = 1.0
volume_mad_threshold = 1.0

# 价格历史分位窗口。
# close_rank_20 用于短期价格强弱判断，close_rank_60 主要进入输出特征和图形，
# min_rank_days 控制早期样本不足时不计算短窗口分位。
price_rank_window = 30
price_rank_long_window = 60
min_rank_days = 8
min_rank_long_days = 20

# 持仓量变化和成交量 MAD 参照窗口。
# oi_change_window=5 表示先计算 5 日持仓变化率；
# oi_change_mad_window=10 表示再用过去 10 天的变化率作为历史参照；
# volume_mad_window=10 表示成交量异常也用过去 10 天做历史参照。
oi_change_window = 5
oi_change_mad_window = 10
volume_mad_window = 10
min_mad_days = 5

# 触发信号后的次一交易日，根据开盘价和前一交易日均值决定方向。
# 这里的均值使用日线 OHLC4 均价，避免依赖不同品种成交额乘数。
# 公共 position 字段保留为 1.0，真实回测方向写入 trade_open_mean_position。
signal_position_scale = 1
signal_holding_days = 1

# 均线排列和乖离率过滤参数。
# ma5 > ma10 > ma20 用于确认短中期多头排列；
# ma_gap_threshold 要求 ma5/ma10、ma10/ma20 之间至少拉开一定距离。
ma_short_window = 5
ma_mid_window = 10
ma_long_window = 20
ma_gap_threshold = 0.01


# =========================
# 1. 读取日频数据
# =========================

daily = load_daily(symbol)

# =========================
# 2. 计算收益和价格历史分位
# =========================
# daily_return 是 1 日收益率，ret_5 是 5 日收益率。
# past_rank() 使用今天以前的历史样本比较当前价格，
# 因此 close_rank_20/60 表示今天收盘价在过去窗口中的相对位置。

daily["daily_return"] = daily["close"].pct_change()
daily["ret_5"] = daily["close"].pct_change(5)

daily["close_rank_20"] = past_rank(
    daily["close"],
    window=price_rank_window,
    min_history_days=min_rank_days,
)
daily["close_rank_60"] = past_rank(
    daily["close"],
    window=price_rank_long_window,
    min_history_days=min_rank_long_days,
)

# =========================
# 3. 计算成交量异常：MAD 方法
# =========================
# 成交量先过滤非正值再取 log，避免对 0 或负数取对数。
# volume_mad_score 衡量今天 log(volume) 相对过去窗口中位数的偏离程度；
# MAD 方法比均值和标准差更稳健，不容易被极端值影响。

safe_volume = daily["volume"].where(daily["volume"] > 0, np.nan)
daily["log_volume"] = np.log(safe_volume)

(
    daily["log_volume_median_past"],
    daily["log_volume_mad_past"],
    daily["volume_mad_score"],
) = mad_score(
    daily["log_volume"],
    window=volume_mad_window,
    min_history_days=min_mad_days,
)

# =========================
# 4. 计算持仓变化异常：MAD 方法
# =========================
# OI 变化率的 MAD score：
# 使用 5 日持仓量变化率，并和过去 oi_change_mad_window 天比较。
# 注意：后续信号使用的是 MAD score 的绝对值，
# 也就是只判断持仓变化是否异常剧烈，不在这里限定增加或下降方向。

daily["oi_change_5"] = (
    daily["open_interest"] -
    daily["open_interest"].shift(oi_change_window)
)
daily["oi_change_5_rate"] = (
    daily["oi_change_5"] /
    daily["open_interest"].shift(oi_change_window).replace(0, np.nan)
)

(
    daily["oi_change_5_rate_median_past"],
    daily["oi_change_5_rate_mad_past"],
    daily["oi_change_5_rate_mad_score"],
) = mad_score(
    daily["oi_change_5_rate"],
    window=oi_change_mad_window,
    min_history_days=min_mad_days,
)

daily["oi_change_5_rate_mad_abs_score"] = (
    daily["oi_change_5_rate_mad_score"].abs()
)

# =========================
# 5. 计算价格、成交量、持仓变化综合得分
# =========================
# price_volume_oi_score 是连续因子值：
# 价格分位越高、5 日收益越强、持仓变化越异常、成交量越放大，得分越高。
# positive_part() 只保留正向贡献，负向或缺失值不会降低综合得分。

daily["price_volume_oi_score"] = (
    daily["close_rank_20"].fillna(0)
    + positive_part(daily["ret_5"])
    + positive_part(daily["oi_change_5_rate_mad_abs_score"])
    + positive_part(daily["volume_mad_score"])
)

# =========================
# 5.1 计算均线排列和乖离率过滤
# =========================
# ma_bull_stack_filter 要求价格均线形成 ma5 > ma10 > ma20 的多头排列，
# 并且 ma5/ma10、ma10/ma20 的乖离率均超过 ma_gap_threshold。

daily["ma5"] = (
    daily["close"]
    .rolling(ma_short_window, min_periods=ma_short_window)
    .mean()
)
daily["ma10"] = (
    daily["close"]
    .rolling(ma_mid_window, min_periods=ma_mid_window)
    .mean()
)
daily["ma20"] = (
    daily["close"]
    .rolling(ma_long_window, min_periods=ma_long_window)
    .mean()
)

daily["ma5_ma10_bias"] = daily["ma5"] / daily["ma10"] - 1
daily["ma10_ma20_bias"] = daily["ma10"] / daily["ma20"] - 1
daily["close_ma20_bias"] = daily["close"] / daily["ma20"] - 1

daily["ma_bull_stack_filter"] = (
    (daily["ma5"] > daily["ma10"]) &
    (daily["ma10"] > daily["ma20"]) &
    (daily["ma5_ma10_bias"] >= ma_gap_threshold) &
    (daily["ma10_ma20_bias"] >= ma_gap_threshold)
)

# =========================
# 6. 生成价格上涨、成交放大、持仓变化异常信号
# =========================
# 信号要求五类条件同时成立：
# 1. 收盘价处在近期高分位；
# 2. 过去 5 日价格上涨；
# 3. 5 日持仓变化率相对历史窗口明显异常；
# 4. 成交量相对历史窗口明显放大。
# 5. 均线呈 ma5 > ma10 > ma20 多头排列，且短中期均线乖离率达标。
# 这里捕捉的是“持仓变化异常”，不是单纯的持仓下降。

daily["price_up_volume_oi_surge_signal"] = 0
daily.loc[
    (
        daily["close_rank_20"] >= price_rank_threshold
    ) & (
        daily["ret_5"] > 0
    ) & (
        daily["oi_change_5_rate_mad_abs_score"] >= oi_change_mad_threshold
    ) & (
        daily["volume_mad_score"] >= volume_mad_threshold
    ) & (
        daily["ma_bull_stack_filter"]
    ),
    "price_up_volume_oi_surge_signal",
] = 1

# =========================
# 7. 生成信号次日开盘方向
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

signal_mask = daily["price_up_volume_oi_surge_signal"] == 1
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
    daily["price_up_volume_oi_surge_signal"]
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
# 8. 保存标准化结果
# =========================
# factor_value 使用 price_volume_oi_score，signal 使用二值信号列。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会统一生成 factor_id、factor_name、signal、position、position_scale。
# backtest_sum.py 会优先使用 trade_open_mean_position 作为交易当天仓位。

feature_columns = [
    "daily_return",
    "ret_5",
    "close_rank_20",
    "close_rank_60",
    "oi_change_5",
    "oi_change_5_rate",
    "oi_change_5_rate_median_past",
    "oi_change_5_rate_mad_past",
    "oi_change_5_rate_mad_score",
    "oi_change_5_rate_mad_abs_score",
    "log_volume",
    "log_volume_median_past",
    "log_volume_mad_past",
    "volume_mad_score",
    "price_volume_oi_score",
    "ma5",
    "ma10",
    "ma20",
    "ma5_ma10_bias",
    "ma10_ma20_bias",
    "close_ma20_bias",
    "ma_bull_stack_filter",
    "day_mean_price",
    "next_day_open",
    "next_day_open_vs_day_mean",
    "signal_next_day_position",
    "previous_day_mean_price",
    "open_vs_previous_day_mean",
    "trade_from_previous_signal",
    "trade_open_mean_position",
]

save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="price_volume_oi_score",
    signal_column="price_up_volume_oi_surge_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "close_rank_20",
        "ret_5",
        "oi_change_5_rate_mad_abs_score",
        "volume_mad_score",
        "ma5_ma10_bias",
        "ma10_ma20_bias",
    ],
    signal_holding_days=signal_holding_days,
)
