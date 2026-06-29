# 因子乖离率计算笔记

## 1. 这里的乖离率是什么

本项目里的乖离率主要用于衡量价格均线之间是否真正拉开距离，而不是只满足一个很弱的 `ma5 > ma10 > ma20` 多头排列。

通用公式：

```text
bias(a, b) = a / b - 1
```

如果写成百分比：

```text
bias_pct(a, b) = (a / b - 1) * 100%
```

例如 `ma5_ma10_bias = 0.012`，表示 5 日均线比 10 日均线高 1.2%。

## 2. 当前因子使用的三个乖离率

以收盘价 `close` 计算 5 日、10 日、20 日移动平均：

```python
ma5 = close.rolling(5, min_periods=5).mean()
ma10 = close.rolling(10, min_periods=10).mean()
ma20 = close.rolling(20, min_periods=20).mean()
```

然后计算：

```python
ma5_ma10_bias = ma5 / ma10 - 1
ma10_ma20_bias = ma10 / ma20 - 1
close_ma20_bias = close / ma20 - 1
```

含义：

- `ma5_ma10_bias`：短期均线相对中期均线的强度。
- `ma10_ma20_bias`：中期均线相对长期均线的强度。
- `close_ma20_bias`：当前收盘价相对 20 日均线的位置，适合观察价格是否过热或偏离趋势中枢。

## 3. 多头排列 + 乖离率过滤

当前代码里的核心过滤条件是：

```python
ma_gap_threshold = 0.01

ma_bull_stack_filter = (
    (ma5 > ma10)
    & (ma10 > ma20)
    & (ma5_ma10_bias >= ma_gap_threshold)
    & (ma10_ma20_bias >= ma_gap_threshold)
)
```

也就是：

- `ma5 > ma10 > ma20`：价格处在短中期上行趋势。
- `ma5 / ma10 - 1 >= 1%`：短期均线至少比中期均线高 1%。
- `ma10 / ma20 - 1 >= 1%`：中期均线至少比长期均线高 1%。

`close_ma20_bias` 目前主要作为输出特征和观察指标，没有直接进入过滤条件。后续如果想控制过热，可以加类似条件：

```python
close_ma20_bias <= 0.08
```

表示收盘价相对 20 日均线不要高出 8% 以上。

## 4. 四个因子的对应关系

### 11 price_volume_up_oi_down

当前字段：

```python
ma5
ma10
ma20
ma5_ma10_bias
ma10_ma20_bias
close_ma20_bias
ma_bull_stack_filter
```

信号在原有价格高分位、5 日收益为正、持仓变化异常、成交量放大的基础上，再要求 `ma_bull_stack_filter` 为真。

### 12 price_up_speculation_up

当前字段：

```python
price_ma_5
price_ma_10
price_ma_20
ma5_ma10_bias
ma10_ma20_bias
close_ma20_bias
ma_bull_stack_filter
```

信号要求价格均线多头且乖离率达标，持仓量短期均线强于中期均线，同时投机度 `speculation_mad_score` 达标。

### 13 price_up_oi_down

当前字段：

```python
price_ma_5
price_ma_10
price_ma_20
ma5_ma10_bias
ma10_ma20_bias
close_ma20_bias
ma_bull_stack_filter
```

信号要求价格均线多头且乖离率达标，持仓量处在近期高位，并且当日持仓量相对上一交易日下降。

### 14 uptrend_crowded_chase

当前字段：

```python
ma5
ma10
ma20
ma5_ma10_bias
ma10_ma20_bias
close_ma20_bias
ma_bull_stack_filter
```

信号在价格高分位、5 日上涨、5 日持仓增加、成交量或振幅异常的基础上，再要求均线多头且乖离率达标。

## 5. 参数选择建议

`ma_gap_threshold = 0.01` 是一个偏稳健的默认值，表示短中期趋势至少要有 1% 的均线间距。

可以按研究目的调整：

- `0.005`：更宽松，信号更多，适合先扩样本。
- `0.01`：当前默认，过滤掉刚刚形成但力度不足的多头排列。
- `0.015` 到 `0.02`：更严格，只保留趋势更强、拥挤或过热更明显的阶段。

建议不要只看单品种结果，最好比较全品种的：

- 信号数量是否过少。
- 信号后收益分布是否改善。
- 最大回撤和胜率是否稳定。
- 不同板块是否需要不同阈值。

## 6. 时点和避免未来函数

均线和乖离率使用当日收盘价计算，所以信号只能在当日收盘后确认。

回测或实盘映射时，应使用下一交易日仓位：

```python
trade_from_previous_signal = signal.shift(1)
```

本项目因子里已经用 `trade_from_previous_signal` 和 `trade_open_mean_position` 表示从前一交易日信号推导出的交易日仓位。

## 7. 最小可复用代码

```python
ma_short_window = 5
ma_mid_window = 10
ma_long_window = 20
ma_gap_threshold = 0.01

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
    (daily["ma5"] > daily["ma10"])
    & (daily["ma10"] > daily["ma20"])
    & (daily["ma5_ma10_bias"] >= ma_gap_threshold)
    & (daily["ma10_ma20_bias"] >= ma_gap_threshold)
)
```

如果某个因子里字段名是 `price_ma_5`、`price_ma_10`、`price_ma_20`，公式完全一样，只是把 `ma5`、`ma10`、`ma20` 替换成对应字段名。
