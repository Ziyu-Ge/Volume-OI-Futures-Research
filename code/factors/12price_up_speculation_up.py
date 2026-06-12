from volume_price_factor_utils import (
    SYMBOL,
    load_daily,
    mad_score,
    parse_factor_script_metadata,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = SYMBOL

factor_id, factor_name = parse_factor_script_metadata(__file__)

# 价格均线多头排列窗口。
# ma5 > ma10 > ma20 表示短期价格均线高于中期，中期高于长期；
# 这是原因子里判断“价格已经偏强”的第一层条件。
ma_short_window = 5
ma_mid_window = 10
ma_long_window = 20

# 持仓量均线窗口。
# open_interest_ma_5 > open_interest_ma_10 表示短期持仓量高于中期持仓量；
# 这是原因子里判断“资金参与度也偏强”的第二层条件。
open_interest_short_window = 5
open_interest_mid_window = 10

# 过去多少天作为投机度历史参照。
# MAD 相关的历史中位数和历史 MAD 都只使用今天以前的数据，不包含今天；
# 这样今天的投机度只是待判断对象，不会进入自己的基准样本。
spec_window = 10

# 投机度突然升高的阈值。
# MAD score >= 1 表示今天投机度相对过去一段时间明显偏高；
# 最终信号要求价格均线偏强、持仓量均线偏强、投机度 MAD score 达标三者同时成立。
spec_mad_threshold = 1

# MAD 缩放系数和极小值保护沿用公共 mad_score() 的默认值：
# 1.4826 把 MAD 调整到类似标准差的尺度；
# 1e-12 只是在分母上做极小值保护，同时 MAD <= 0 时仍会被设为 NaN。
mad_scale = 1.4826
mad_epsilon = 1e-12

# 最小历史天数。
# 当过去有效投机度样本少于该值时，不计算 MAD score，避免早期样本太短导致误判。
min_history_days = 5

# 触发信号后，建议仓位比例。
# 没有信号时公共输出函数保持 1.0；有信号时改为这里的 0。
signal_position_scale = 0


# =========================
# 1. 读取日频数据
# =========================

daily = load_daily(symbol)


# =========================
# 2. 计算价格和持仓量均线条件
# =========================
# 注意：价格均线包含今天的收盘价，持仓量均线包含今天的持仓量。
# 这和原实现一致。若之后用于真实交易或回测，仍应在回测端使用 signal.shift(1)
# 产生下一交易日仓位，避免用当天收盘后才确认的信号交易当天。

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

daily["open_interest_ma_5"] = (
    daily["open_interest"]
    .rolling(
        window=open_interest_short_window,
        min_periods=open_interest_short_window,
    )
    .mean()
)
daily["open_interest_ma_10"] = (
    daily["open_interest"]
    .rolling(
        window=open_interest_mid_window,
        min_periods=open_interest_mid_window,
    )
    .mean()
)

daily["is_open_interest_ma_bullish"] = (
    daily["open_interest_ma_5"] > daily["open_interest_ma_10"]
).astype(int)


# =========================
# 3. 计算投机度突然升高：MAD 方法
# =========================
# mad_score() 等价于原来的手写逻辑：
# 1. speculation_median_past 是过去 spec_window 天投机度中位数，不包含今天；
# 2. speculation_mad_past 是过去 spec_window 天投机度的 MAD，不包含今天；
# 3. speculation_mad_score = (今天投机度 - 历史中位数) / (1.4826 * 历史 MAD + 1e-12)；
# 4. 当历史 MAD <= 0 时，MAD score 设为 NaN，避免无波动窗口导致分数失真。
# MAD 比均值和标准差更稳健，不容易被极端值影响。

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
# 这个因子关注价格和持仓量同时偏强时的投机度升高：
# 价格均线呈多头排列，持仓量短期均线高于中期均线，同时投机度突然升高。
# 直觉上，这代表价格已经上涨一段，持仓量和投机度同时升高，
# 可能是跟风资金集中进入，行情进入过热阶段。

daily["price_up_speculation_up_signal"] = (
    (daily["is_ma_bullish"] == 1) &
    (daily["is_open_interest_ma_bullish"] == 1) &
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
    "price_ma_5",
    "price_ma_10",
    "price_ma_20",
    "is_ma_bullish",
    "open_interest",
    "open_interest_ma_5",
    "open_interest_ma_10",
    "is_open_interest_ma_bullish",
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
    signal_column="price_up_speculation_up_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "price_ma_5",
        "price_ma_10",
        "price_ma_20",
        "speculation_mad_score",
    ],
)


# =========================
# 6. 补充因子参数到汇总表
# =========================
# 公共函数已经生成整体汇总表；这里补上本因子特有参数和原汇总中关注的统计量。
# 这样之后只看 summary 文件，也能知道价格均线窗口、持仓量均线窗口、
# 投机度 MAD 窗口、MAD 阈值，以及三类条件分别出现了多少天。

summary_table = result["summary_table"].copy()
summary_table["ma_short_window"] = ma_short_window
summary_table["ma_mid_window"] = ma_mid_window
summary_table["ma_long_window"] = ma_long_window
summary_table["open_interest_short_window"] = open_interest_short_window
summary_table["open_interest_mid_window"] = open_interest_mid_window
summary_table["spec_window"] = spec_window
summary_table["spec_mad_threshold"] = spec_mad_threshold
summary_table["mad_scale"] = mad_scale
summary_table["mad_epsilon"] = mad_epsilon
summary_table["min_history_days"] = min_history_days
summary_table["ma_bullish_days"] = int(daily["is_ma_bullish"].sum())
summary_table["ma_bullish_ratio"] = daily["is_ma_bullish"].mean()
summary_table["open_interest_ma_bullish_days"] = int(
    daily["is_open_interest_ma_bullish"].sum()
)
summary_table["open_interest_ma_bullish_ratio"] = daily[
    "is_open_interest_ma_bullish"
].mean()
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
