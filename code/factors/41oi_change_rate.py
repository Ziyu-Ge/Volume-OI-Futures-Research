import numpy as np

from volume_price_factor_utils import (
    load_daily,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = "LC"

factor_id = "41"
factor_name = "oi_change_rate"

# 持仓量变化率阈值。
# 当前阈值为 0，含义是：只要今天持仓量相对上一交易日下降或不增长，
# 即 oi_change_rate <= 0，就触发减仓信号。
# 如果之后想更严格，可以把阈值改成 -2，表示持仓量下降超过 2% 才触发。
oi_change_rate_threshold = 0

# 单因子触发后，建议仓位比例。
signal_position_scale = 0


# =========================
# 1. 读取日频数据
# =========================

daily = load_daily(symbol)


# =========================
# 2. 计算持仓量变化率因子
# =========================
# 持仓量变化率使用百分比单位：
#   oi_change_rate = (今日持仓量 - 昨日持仓量) / 昨日持仓量 * 100
# 这里的分母必须是上一交易日持仓量，而不是今天持仓量；
# 这样数值表达的是“从昨天到今天，持仓规模变化了多少百分比”。

previous_open_interest = daily["open_interest"].shift(1)

daily["oi_change_rate"] = (
    (daily["open_interest"] - previous_open_interest) /
    previous_open_interest *
    100
)

# 避免上一日持仓量为 0 或负数时出现无意义的变化率。
# 第一行上一日持仓量是 NaN，计算结果自然也是 NaN；
# 这里额外处理 <= 0 的情况，和原实现保持一致。
daily.loc[
    previous_open_interest <= 0,
    "oi_change_rate",
] = np.nan


# =========================
# 3. 生成持仓量下降信号
# =========================
# 如果持仓量变化率 <= 阈值，则认为持仓资金正在退出或至少没有继续增加，
# 触发减仓信号。这里保留 <=，不改成 <；
# 因为阈值为 0 时，持仓量刚好不变也会被视为信号。

daily["oi_change_rate_signal"] = (
    daily["oi_change_rate"] <= oi_change_rate_threshold
).astype(int)


# =========================
# 4. 保存标准化结果
# =========================
# factor_value 仍然使用 oi_change_rate，和原实现一致。
# feature_columns 会进入因子每日表、信号事件表和汇总表；
# 公共函数会统一生成 factor_id、factor_name、signal、position、position_scale，
# 并负责保存 CSV 与基础图形输出。

feature_columns = [
    "open_interest",
    "oi_change_rate",
]

result = save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="oi_change_rate",
    signal_column="oi_change_rate_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "oi_change_rate",
    ],
)


# =========================
# 5. 补充因子参数到汇总表
# =========================
# 公共函数已经生成整体汇总表；这里补上本因子特有参数和原汇总中关注的统计量。
# 这样之后只看 summary 文件，也能知道触发阈值、起止持仓量、
# 以及持仓量变化率和持仓量自身的大致分布。

summary_table = result["summary_table"].copy()
summary_table["oi_change_rate_threshold"] = oi_change_rate_threshold
summary_table["mean_oi_change_rate"] = daily["oi_change_rate"].mean()
summary_table["max_oi_change_rate"] = daily["oi_change_rate"].max()
summary_table["min_oi_change_rate"] = daily["oi_change_rate"].min()
summary_table["start_open_interest"] = daily["open_interest"].iloc[0]
summary_table["end_open_interest"] = daily["open_interest"].iloc[-1]
summary_table["mean_open_interest"] = daily["open_interest"].mean()
summary_table["max_open_interest"] = daily["open_interest"].max()
summary_table["min_open_interest"] = daily["open_interest"].min()
summary_table.to_csv(result["summary_output_path"], index=False)
