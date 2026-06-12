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
price_rank_threshold = 0.70
oi_change_mad_threshold = 1.0
volume_mad_threshold = 1.0

# 价格历史分位窗口。
# close_rank_20 用于短期价格强弱判断，close_rank_60 主要进入输出特征和图形，
# min_rank_days 控制早期样本不足时不计算短窗口分位。
price_rank_window = 20
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

# 触发信号后，建议仓位比例和持有天数。
# signal_holding_days 会在公共输出函数中把 position 影响延续 5 天；
# signal 列本身仍只记录当天是否触发原始信号。
signal_position_scale = 2
signal_holding_days = 5


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
# 6. 生成价格上涨、成交放大、持仓变化异常信号
# =========================
# 信号要求四类条件同时成立：
# 1. 收盘价处在近期高分位；
# 2. 过去 5 日价格上涨；
# 3. 5 日持仓变化率相对历史窗口明显异常；
# 4. 成交量相对历史窗口明显放大。
# 虽然 factor_name 保留了 oi_down 命名，但当前代码实际使用绝对值条件，
# 因此这里捕捉的是“持仓变化异常”，不是单纯的持仓下降。

daily["price_volume_up_oi_down_signal"] = 0
daily.loc[
    (
        daily["close_rank_20"] >= price_rank_threshold
    ) & (
        daily["ret_5"] > 0
    ) & (
        daily["oi_change_5_rate_mad_abs_score"] >= oi_change_mad_threshold
    ) & (
        daily["volume_mad_score"] >= volume_mad_threshold
    ),
    "price_volume_up_oi_down_signal",
] = 1

# =========================
# 7. 保存标准化结果
# =========================
# factor_value 使用 price_volume_oi_score，signal 使用二值信号列。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会统一生成 factor_id、factor_name、signal、position、position_scale，
# 并负责保存 CSV 与基础图形输出。

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
]

save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="price_volume_oi_score",
    signal_column="price_volume_up_oi_down_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "close_rank_20",
        "ret_5",
        "oi_change_5_rate_mad_abs_score",
        "volume_mad_score",
    ],
    signal_holding_days=signal_holding_days,
)
