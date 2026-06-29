import numpy as np

from volume_price_factor_utils import (
    SYMBOL,
    add_volume_price_features,
    load_daily,
    parse_factor_script_metadata,
    positive_part,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = SYMBOL

factor_id, factor_name = parse_factor_script_metadata(__file__)

# 价格相对过去 20 个交易日的历史分位阈值。
# close_rank_20 >= 0.9 表示今天收盘价高于过去窗口中至少 90% 的收盘价，
# 用来确认价格已经处在近期偏强区域。
price_rank_threshold = 0.9

# 5 日持仓量对数变化阈值。
# oi_ret_5 > 0.05 表示最近 5 日持仓量明显增加，
# 对应“上涨过程中资金/持仓继续追入”的条件。
oi_ret_5_threshold = 0.05

# 成交量和日内振幅的 MAD score 阈值。
# volume_mad_score 或 range_mad_score 达标即可，
# 表示追涨过程中至少出现成交放大或波动放大的拥挤特征。
volume_mad_threshold = 1.2
range_mad_threshold = 1.2

# 收盘价在当日高低点区间中的位置上限。
# close_location = (close - low) / (high - low)；
# <= 0.95 沿用原逻辑，避免只捕捉几乎收在最高点的极端 K 线。
close_location_threshold = 0.95

# 触发信号后的次一交易日，根据开盘价和前一交易日均值决定方向。
# 这里的均值使用日线 OHLC4 均价，避免依赖不同品种成交额乘数。
# 公共 position 字段保留为 1.0，真实回测方向写入 trade_open_mean_position。
signal_position_scale = 1

# 均线排列和乖离率过滤参数。
# ma5 > ma10 > ma20 用于确认短中期多头排列；
# ma_gap_threshold 要求 ma5/ma10、ma10/ma20 之间至少拉开一定距离。
ma_short_window = 5
ma_mid_window = 10
ma_long_window = 20
ma_gap_threshold = 0.015

# =========================
# 1. 读取日频数据并补充量价特征
# =========================
# add_volume_price_features() 会统一计算收益、价格历史分位、
# 持仓量/成交量/振幅的 MAD score，以及 close_location 等公共字段。

daily = load_daily(symbol)
daily = add_volume_price_features(daily)

# =========================
# 2. 计算拥挤追涨得分
# =========================
# crowded_chase_score 是连续因子值：
# 价格分位越高、持仓量异常增加越明显，得分越高；
# 成交量和振幅异常也会抬高得分，但各自只给 0.5 权重。
# positive_part() 只保留正向异常，负向或缺失的 MAD score 不降低得分。

daily["crowded_chase_score"] = (
    daily["close_rank_20"].fillna(0)
    + positive_part(daily["open_interest_mad_score"])
    + 0.5 * positive_part(daily["volume_mad_score"])
    + 0.5 * positive_part(daily["range_mad_score"])
)

# =========================
# 2.1 计算均线排列和乖离率过滤
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
# 3. 生成上行趋势拥挤追涨信号
# =========================
# 信号要求六类条件同时成立：
# 1. 收盘价处在近期高分位；
# 2. 过去 5 日价格上涨；
# 3. 过去 5 日持仓量增加超过阈值；
# 4. 成交量或振幅至少一项出现明显正向异常；
# 5. 收盘价位置不高于 close_location_threshold。
# 6. 均线呈 ma5 > ma10 > ma20 多头排列，且短中期均线乖离率达标。
# 这些条件合在一起刻画“上涨后继续追入且交易变拥挤”的阶段。

daily["uptrend_crowded_chase_signal"] = 0
daily.loc[
    (
        daily["close_rank_20"] >= price_rank_threshold
    ) & (
        daily["ret_5"] > 0
    ) & (
        daily["oi_ret_5"] > oi_ret_5_threshold
    ) & (
        (daily["volume_mad_score"] >= volume_mad_threshold) |
        (daily["range_mad_score"] >= range_mad_threshold)
    ) & (
        daily["close_location"] <= close_location_threshold
    ) & (
        daily["ma_bull_stack_filter"]
    ),
    "uptrend_crowded_chase_signal",
] = 1

# =========================
# 4. 生成信号次日开盘方向
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

signal_mask = daily["uptrend_crowded_chase_signal"] == 1
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
    daily["uptrend_crowded_chase_signal"]
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
# 5. 保存标准化结果
# =========================
# factor_value 使用 crowded_chase_score，signal 使用二值信号列。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会统一生成 factor_id、factor_name、signal、position、position_scale。
# backtest_sum.py 会优先使用 trade_open_mean_position 作为交易当天仓位。

feature_columns = [
    "daily_return",
    "ret_5",
    "close_rank_20",
    "close_rank_60",
    "oi_ret_5",
    "open_interest_mad_score",
    "volume_mad_score",
    "range_pct",
    "range_mad_score",
    "close_location",
    "crowded_chase_score",
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
    factor_value_column="crowded_chase_score",
    signal_column="uptrend_crowded_chase_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "close_rank_20",
        "close_rank_60",
        "open_interest_mad_score",
        "volume_mad_score",
        "range_mad_score",
        "ma5_ma10_bias",
        "ma10_ma20_bias",
    ],
)
