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
# close_rank_20 >= 0.75 表示今天收盘价高于过去窗口中至少 75% 的收盘价，
# 用来确认价格已经处在近期偏强区域。
price_rank_threshold = 0.75 

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

# 触发信号后，建议仓位比例。
signal_position_scale = 0


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
# 3. 生成上行趋势拥挤追涨信号
# =========================
# 信号要求五类条件同时成立：
# 1. 收盘价处在近期高分位；
# 2. 过去 5 日价格上涨；
# 3. 过去 5 日持仓量增加超过阈值；
# 4. 成交量或振幅至少一项出现明显正向异常；
# 5. 收盘价位置不高于 close_location_threshold。
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
    ),
    "uptrend_crowded_chase_signal",
] = 1

# =========================
# 4. 保存标准化结果
# =========================
# factor_value 使用 crowded_chase_score，signal 使用二值信号列。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会统一生成 factor_id、factor_name、signal、position、position_scale，
# 并负责保存 CSV 与基础图形输出。

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
    ],
)
