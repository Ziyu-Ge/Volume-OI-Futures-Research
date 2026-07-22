# chapter2 21-24 号因子和 rolling 因子筛选逻辑

本文只说明筛选逻辑和公式，不展开回测收益、图表和代码执行细节。

## 1. 统一公式

设第 `d` 个交易日：

- `C_d`：收盘价。
- `H_d`：最高价。
- `L_d`：最低价。
- `V_d`：成交量。
- `OI_d`：持仓量。

### 均线

`n` 日均线：

```text
MA(n,d) = 最近 n 个交易日收盘价的平均值
```

两个乖离差：

```text
B1_d = C_d / MA(long,d)  - C_d / MA(short,d)
B2_d = C_d / MA(trend,d) - C_d / MA(long,d)
```

直观理解：

- `B1` 看短均线和中均线之间的偏离。
- `B2` 看中均线和长均线之间的偏离。

### 回归斜率

对任意序列 `X`，在最近 `n` 日做一条直线，斜率记为：

```text
beta(X,n,d)
```

代码里价格和持仓用归一化斜率：

```text
R(X,n,d) = beta(X,n,d) / 最近 n 日 X 的平均值
```

所以：

```text
R(OI,n,d) = 持仓量斜率 / 最近 n 日平均持仓量
R(C,n,d)  = 收盘价斜率 / 最近 n 日平均收盘价
```

### 投机度

投机度：

```text
S_d = log(V_d / OI_d)
```

投机度回落条件直接用斜率：

```text
beta(S,n,d) <= 阈值
```

### 波动率

平仓用最近 `m` 个已完成交易日的平均日内振幅率，不包含当天：

```text
vol_d = mean((H_{d-i} - L_{d-i}) / C_{d-i}), i = 1,2,...,m
```

## 2. 开空筛选逻辑

所有条件同时满足，才产生开空信号。任一指标历史不足或为空，就不开空。

通用公式：

```text
open_short_d = 1(
       B1_d >= B1下限
   and B2_d >= B2下限
   and B2_d <= B2上限, 如果该因子设置了上限
   and R(OI, oi窗口, d) <= 持仓斜率阈值
   and R(C,  close窗口, d) <= 价格斜率阈值
   and beta(S, 投机度窗口, d) <= 投机度斜率阈值, 如果该因子启用投机度
)
```

### 21 号因子

名称：`high_bias_oi_drop`

筛选含义：价格偏离均线较高，同时持仓和价格开始回落。

```text
B1_d >= 0.04
B2_d >= 0.10
R(OI, 7, d) <= 0
R(C,  7, d) <= 0
```

21 号不限制 `B2` 上限，也不看投机度。

### 22 号因子

名称：`high_bias_oi_drop_mixed`

筛选含义：和 21 号接近，但持仓回落看得更长，并限制长期乖离不能太高。

```text
B1_d >= 0.04
0.10 <= B2_d <= 0.18
R(OI, 15, d) <= 0
R(C,   7, d) <= 0
```

22 号不看投机度。

### 23 号因子

名称：`high_bias_oi_speculation_drop`

筛选含义：在 21 号/22 号的高乖离、持仓回落、价格回落基础上，再要求投机度回落。

```text
B1_d >= 0.04
0.10 <= B2_d <= 0.18
R(OI, 5, d) <= 0
R(C,  7, d) <= 0
beta(S, 5, d) <= -0.01
```

### 24 号因子

名称：`high_bias_oi_speculation_drop_mixed`

筛选含义：用更短的均线和更宽松的价格、持仓斜率条件，再要求投机度快速回落。

```text
B1_d >= -0.006
-0.05 <= B2_d <= 0.12
R(OI, 7, d) <= 0.003
R(C,  7, d) <= 0.0065
beta(S, 3, d) <= -0.013
```

注意：24 号的 `B1` 和 `B2` 下限可以为负，所以它不是简单筛“乖离必须很高”，而是筛“短中长均线关系落在指定区间内”。

## 3. 平空筛选逻辑

开空后，记：

- `E`：开仓价。
- `L*_t`：开仓后到当前 bar 为止的最低价。
- `P_t`：当前 bar 收盘价。
- `vol`：当前交易日对应的历史平均振幅率。

平空有两条线。

### 开仓价止损线

如果 `entry_loss_volatility_multiplier = 0`：

```text
entry_stop = E
```

否则：

```text
entry_stop = E * (1 + vol * entry_loss_volatility_multiplier)
```

### 低点反弹线

```text
trailing_stop = L*_t * (1 + vol * trailing_multiplier)
```

### 平空总条件

```text
cover_short_t = 1(
       P_t > entry_stop
    or P_t > trailing_stop
)
```

21/23 用日线收盘价检查平空；22/24 用小时线收盘价检查平空。

各因子平仓参数：

| 因子 | trailing_multiplier | entry_loss_volatility_multiplier |
| --- | ---: | ---: |
| 21 | 4.0 | 0 |
| 22 | 3.5 | 0 |
| 23 | 4.0 | 0 |
| 24 | 1.215 | 0.83 |

## 4. 21-24 号因子参数总表

| 因子 | 执行频率 | short/long/trend | B1 下限 | B2 下限 | B2 上限 | 持仓斜率 | 价格斜率 | 投机度斜率 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 21 | 日频 | 5/20/60 | 0.04 | 0.10 | 无 | `R(OI,7)<=0` | `R(C,7)<=0` | 不使用 |
| 22 | 小时 | 5/20/60 | 0.04 | 0.10 | 0.18 | `R(OI,15)<=0` | `R(C,7)<=0` | 不使用 |
| 23 | 日频 | 5/20/60 | 0.04 | 0.10 | 0.18 | `R(OI,5)<=0` | `R(C,7)<=0` | `beta(S,5)<=-0.01` |
| 24 | 小时 | 2/5/7 | -0.006 | -0.05 | 0.12 | `R(OI,7)<=0.003` | `R(C,7)<=0.0065` | `beta(S,3)<=-0.013` |

## 5. rolling 因子筛选逻辑

rolling 因子名称：`factor_24_walk_forward`

rolling 不发明新公式。它仍然使用 24 号因子的开空和平空公式，只是每隔一段时间重新选择参数。

### 滚动窗口

```text
训练期 = 过去 3 年
测试期 = 之后 6 个月
每 6 个月向前滚动一次
```

### 参数怎么选

每个训练窗口里，代码会试一批候选参数。每个候选参数都跑一遍训练期回测，然后按下面分数排序：

```text
score = Sharpe - 2.0 * abs(max_drawdown)
```

选择 `score` 最高的参数，拿去跑下一个 6 个月测试期。

### 候选参数

候选参数不是全组合搜索，而是：

```text
基准 24 号参数
+ 每次只改一个字段的参数
```

可选值：

| 参数 | 可选值 |
| --- | --- |
| `ma_short` | 2, 3 |
| `ma_long` | 5, 7 |
| `ma_trend` | 7, 10 |
| `B1下限` | -0.006, -0.003 |
| `B2下限` | -0.05, -0.03 |
| `B2上限` | 0.12, 0.10 |
| `oi_slope_window` | 7, 10 |
| `oi_slope_threshold` | 0.003, 0.001 |
| `close_slope_window` | 7, 10 |
| `close_slope_threshold` | 0.0065, 0.004 |
| `speculation_slope_window` | 3, 5 |
| `speculation_slope_threshold` | -0.013, -0.010 |
| `volatility_window` | 10, 15 |
| `trailing_multiplier` | 1.215, 1.40 |
| `entry_loss_volatility_multiplier` | 0.83, 1.00 |

一句话总结：

```text
rolling 因子 = 用过去 3 年表现筛出一套 24 号参数，再用这套参数交易未来 6 个月。
```

## 6. 回测结果展示

下表使用各因子 `symbol_metrics.csv` 中的 `ALL_SYMBOLS_EQUAL_WEIGHT` 组合行。交易次数使用对应 `trades.csv` 的总交易行数，包含期末仍未平仓的交易。

| 因子 | 年化收益率 | 最大回撤 | 夏普 | 交易次数 |
| --- | ---: | ---: | ---: | ---: |
| 21 | 6.26% | -23.13% | 0.55 | 124 |
| 22 | 9.53% | -22.60% | 0.70 | 95 |
| 23 | 5.90% | -23.66% | 0.51 | 62 |
| 24 | 17.04% | -6.42% | 1.67 | 9352 |
| 24 rolling | 10.18% | -9.64% | 1.23 | 8064 |

### 21 号因子 summary

| return summary | all symbols summary |
| --- | --- |
| ![21 号因子 return summary](../useful_plots/chapter2_factor_21_return_summary.png) | ![21 号因子 all symbols summary](../useful_plots/chapter2_factor_21_all_symbols_summary.png) |

### 22 号因子 summary

| return summary | all symbols summary |
| --- | --- |
| ![22 号因子 return summary](../useful_plots/chapter2_factor_22_return_summary.png) | ![22 号因子 all symbols summary](../useful_plots/chapter2_factor_22_all_symbols_summary.png) |

### 23 号因子 summary

| return summary | all symbols summary |
| --- | --- |
| ![23 号因子 return summary](../useful_plots/chapter2_factor_23_return_summary.png) | ![23 号因子 all symbols summary](../useful_plots/chapter2_factor_23_all_symbols_summary.png) |

### 24 号因子 summary

| return summary | all symbols summary |
| --- | --- |
| ![24 号因子 return summary](../useful_plots/chapter2_factor_24_return_summary.png) | ![24 号因子 all symbols summary](../useful_plots/chapter2_factor_24_all_symbols_summary.png) |

### 24 rolling 因子 summary

| return summary | all symbols summary |
| --- | --- |
| ![24 rolling 因子 return summary](../useful_plots/chapter2_factor_24_rolling_return_summary.png) | ![24 rolling 因子 all symbols summary](../useful_plots/chapter2_factor_24_rolling_all_symbols_summary.png) |
