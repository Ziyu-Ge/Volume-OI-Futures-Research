# 因子逻辑笔记

## 11 price_up_volume_oi_surge

总结：

该算法用于识别商品期货中“**价格处于高位、近期上涨、成交量明显放大、持仓量变化异常**”的交易日。

核心思想是：

当价格已经处在近期偏高位置，同时成交量放大、持仓量剧烈变化，说明**市场交易活跃度和资金分歧明显上升，可能出现短期趋势延续或反转机会**。

### 计算公式：

- `ret_5 = close_t / close_{t-5} - 1`
- `close_rank_20 = mean(close_{t-i} <= close_t), i = 1..30`，最少 8 个历史样本。注：字段名为 `close_rank_20`，脚本实际窗口为 30 日。
- `log_volume = log(volume)`
- `volume_mad_score = (log_volume_t - median(log_volume_{t-10:t-1})) / (1.4826 * MAD(log_volume_{t-10:t-1}) + 1e-12)`，最少 5 个历史样本。
- `oi_change_5 = open_interest_t - open_interest_{t-5}`
- `oi_change_5_rate = oi_change_5 / open_interest_{t-5}`
- `oi_change_5_rate_mad_score = (oi_change_5_rate_t - median(oi_change_5_rate_{t-10:t-1})) / (1.4826 * MAD(oi_change_5_rate_{t-10:t-1}) + 1e-12)`，最少 5 个历史样本。
- `oi_change_5_rate_mad_abs_score = abs(oi_change_5_rate_mad_score)`
- `maN = mean(close_{t-N+1:t})`
- `ma5_ma10_bias = ma5 / ma10 - 1`
- `ma10_ma20_bias = ma10 / ma20 - 1`
- `price_volume_oi_score = fillna(close_rank_20, 0) + max(ret_5, 0) + max(oi_change_5_rate_mad_abs_score, 0) + max(volume_mad_score, 0)`

### 筛选条件：

**价格**

- `ma5 > ma10 > ma20`：要求短中期均线呈多头排列，确认价格趋势方向向上。
- `ma5 > ma120`：要求短期均线高于长期均线，过滤掉长期趋势仍偏弱的反弹。
- `ma5_ma10_bias >= 0.01`：要求 5 日均线相对 10 日均线至少拉开 1%，避免均线刚刚缠绕时误触发。
- `ma10_ma20_bias >= 0.01`：要求 10 日均线相对 20 日均线至少拉开 1%，确认中短期趋势有一定斜率。
- `close_rank_20 >= 0.90`：要求收盘价处在近期高位，确认价格已经进入偏强区域。
- `ret_5 > 0`：要求过去 5 日价格上涨，避免把高位但已经转弱的样本纳入信号。

**成交量**

- `volume_mad_score >= 1.0`：要求成交量相对过去窗口明显放大，确认市场交易活跃度提升。

**持仓量**

- `oi_change_5_rate_mad_abs_score >= 1.0`：要求 5 日持仓变化率相对历史波动明显异常，捕捉持仓层面的剧烈变化。


## 12 price_up_speculation_up

总结：

这个因子把价格多头趋势、**持仓量多头排列**和**投机度异常**升高放在一起判断。构造重点是识别价格已经走强后，短期投机交易进一步升温的阶段。

### 计算公式：

- `speculation = log(volume / open_interest)`
- `maN = mean(close_{t-N+1:t})`
- `ma5_ma10_bias = ma5 / ma10 - 1`
- `ma10_ma20_bias = ma10 / ma20 - 1`
- `open_interest_ma_5 = mean(open_interest_{t-4:t})`
- `open_interest_ma_10 = mean(open_interest_{t-9:t})`
- `speculation_mad_score = (speculation_t - median(speculation_{t-10:t-1})) / (1.4826 * MAD(speculation_{t-10:t-1}) + 1e-12)`，最少 5 个历史样本。
- `factor_value = speculation_mad_score`

### 筛选条件：

**价格**

- `ma5 > ma10 > ma20`：要求短中期价格均线多头排列，确认价格趋势处于上行状态。
- `ma5 > ma120`：要求短期价格强于长期均线，保证中长期趋势背景也偏强。
- `ma5_ma10_bias >= 0.01`：要求 5 日均线高出 10 日均线至少 1%，过滤弱趋势或均线粘合状态。
- `ma10_ma20_bias >= 0.01`：要求 10 日均线高出 20 日均线至少 1%，确认趋势不是短暂扰动。

**持仓量**

- `open_interest_ma_5 > open_interest_ma_10`：要求短期持仓均线高于中期持仓均线，表示近期持仓参与度上升。

**投机度**

- `speculation_mad_score >= 1`：要求投机度相对过去窗口明显偏高，捕捉成交相对持仓快速升温。

## 13 price_up_oi_down

总结：

这个因子刻画价格趋势仍强，但**持仓量在近期高位开始下降**的状态。它更像是在上涨过程中**寻找持仓背离或高位减仓迹象**。

### 计算公式：

- `maN = mean(close_{t-N+1:t})`
- `ma5_ma10_bias = ma5 / ma10 - 1`
- `ma10_ma20_bias = ma10 / ma20 - 1`
- `open_interest_rank_10 = mean(open_interest_{t-i} <= open_interest_t), i = 1..10`，最少 5 个历史样本。
- `log_open_interest = log(open_interest)`
- `delta_log_open_interest = log_open_interest_t - log_open_interest_{t-1}`
- `oi_change_rate = (exp(delta_log_open_interest) - 1) * 100`
- `log_open_interest_mad_score = (log_open_interest_t - median(log_open_interest_{t-10:t-1})) / (1.4826 * MAD(log_open_interest_{t-10:t-1}) + 1e-12)`，最少 5 个历史样本。
- `factor_value = log_open_interest_mad_score`

### 筛选条件：

**价格**

- `ma5 > ma10 > ma20`：要求短中期价格均线多头排列，确认价格仍处在上行趋势中。
- `ma5 > ma120`：要求短期趋势强于长期趋势，避免长期下行中的短线反弹误触发。
- `ma5_ma10_bias >= 0.01`：要求 5 日均线相对 10 日均线有足够正乖离，过滤趋势不清晰的样本。
- `ma10_ma20_bias >= 0.01`：要求 10 日均线相对 20 日均线有足够正乖离，确认中短期趋势强度。

**持仓量**

- `open_interest_rank_10 >= 0.8`：要求持仓量处在过去 10 日高位，说明当前持仓基础仍然偏高。
- `delta_log_open_interest < 0`：要求当天持仓量相对上一日下降，用来识别高位持仓回落。

## 14 uptrend_crowded_chase

总结：

这个因子用价格高位上涨、**持仓继续增加**以及**成交量或振幅异常**来刻画**追涨拥挤**。它关注的是趋势上行后，资金继续追入并伴随交易拥挤的状态。

### 计算公式：

- `ret_5 = close_t / close_{t-5} - 1`
- `log_open_interest = log(open_interest)`
- `log_volume = log(volume)`
- `oi_ret_5 = log_open_interest_t - log_open_interest_{t-5}`
- `range_pct = (high_t - low_t) / close_t`
- `close_location = (close_t - low_t) / (high_t - low_t)`
- `close_rank_20 = mean(close_{t-i} <= close_t), i = 1..20`，最少 8 个历史样本。
- `open_interest_mad_score = (log_open_interest_t - median(log_open_interest_{t-10:t-1})) / (1.4826 * MAD(log_open_interest_{t-10:t-1}) + 1e-12)`，最少 5 个历史样本。
- `volume_mad_score = (log_volume_t - median(log_volume_{t-10:t-1})) / (1.4826 * MAD(log_volume_{t-10:t-1}) + 1e-12)`，最少 5 个历史样本。
- `range_mad_score = (range_pct_t - median(range_pct_{t-10:t-1})) / (1.4826 * MAD(range_pct_{t-10:t-1}) + 1e-12)`，最少 5 个历史样本。
- `maN = mean(close_{t-N+1:t})`
- `ma5_ma10_bias = ma5 / ma10 - 1`
- `ma10_ma20_bias = ma10 / ma20 - 1`
- `crowded_chase_score = fillna(close_rank_20, 0) + max(open_interest_mad_score, 0) + 0.5 * max(volume_mad_score, 0) + 0.5 * max(range_mad_score, 0)`

### 筛选条件：

**价格**

- `ma5 > ma10 > ma20`：要求短中期均线多头排列，确认趋势结构向上。
- `ma5 > ma120`：要求短期均线位于长期均线上方，保证上涨不是长期弱势里的短暂反抽。
- `ma10_ma20_bias >= 0.01`：要求 10 日均线相对 20 日均线至少高 1%，确认中短期均线已经拉开。
- `ma5_ma10_bias >= 0.01`：要求 5 日均线相对 10 日均线至少高 1%，过滤趋势力度不足的情况。
- `close_rank_20 >= 0.9`：要求收盘价处于过去 20 日高位，确认价格已经明显偏强。
- `ret_5 > 0`：要求过去 5 日价格上涨，保证信号发生在上行阶段。
- `close_location <= 0.95`：要求收盘价不要过度贴近当日最高点，避免只捕捉极端收盘 K 线。

**成交量**

- `volume_mad_score >= 1.2` 或 `range_mad_score >= 1.2`：要求成交量或日内振幅至少一项明显异常，确认交易拥挤或波动放大。

**持仓量**

- `oi_ret_5 > 0.05`：要求过去 5 日持仓量对数变化超过 0.05，表示资金或仓位继续追入。

## 结果
网址：
https://ziyu-ge.github.io/Volume-OI-Futures-Research/results/chapter1/combined/figures/combined_factor_signals_dashboard.html
