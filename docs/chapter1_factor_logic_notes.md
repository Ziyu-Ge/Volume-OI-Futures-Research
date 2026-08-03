# Chapter 1 因子逻辑说明

本文按 `code/chapter1/` 的当前实现重写，说明 11—14 号因子的数据口径、公共计算、和信号触发条件。因子参数以代码中的默认值为准；信号胜率口径另见 `docs/chapter1_signal_evaluation_notes.md`。

## 1. 日频数据口径

`00_prepare_data.py` 先按 `datetime` 排序，再按自然日期聚合分钟数据：

```text
open_d           = 当日第一条分钟记录的 open
close_d          = 当日最后一条分钟记录的 close
high_d           = 当日 high 最大值
low_d            = 当日 low 最小值
volume_d         = 当日 volume 合计
total_turnover_d = 当日 total_turnover 合计
open_interest_d  = 当日最后一条分钟记录的 open_interest
```

若日末持仓量 `open_interest_d <= 0`，代码将其设为空值。投机度在日频数据准备阶段计算：

```text
speculation_d = log(volume_d / open_interest_d)
```

各品种日频表默认保存到：

```text
results/chapter1/tables/{SYMBOL}_daily.csv
```

## 2. 公共计算规则

设第 `d` 个交易日的收盘价、最高价、最低价、成交量和持仓量分别为：

```text
C_d, H_d, L_d, V_d, OI_d
```

### 2.1 历史分位

`past_rank()` 用当前值和此前窗口内的历史值比较，历史样本不包含当天：

```text
Rank(X, w, d)
    = mean(1(X_i <= X_d)), i 属于 d 以前最近 w 条记录中的有效样本
```

空值会从历史样本中剔除。有效历史样本少于指定的最小天数，或者当前值为空时，分位为空。

因此：

- 分位越接近 `1`，当前值相对过去窗口越高。
- 相等值按 `历史值 <= 当前值` 计入。
- 当前值只用于被比较，不会进入自己的历史基准。

### 2.2 MAD 稳健异常分数

`mad_score()` 也只使用当天以前的历史样本。对序列 `X`，设过去 `w` 条记录中的有效样本为 `H_d`：

```text
median_past_d = median(H_d)
MAD_past_d    = median(|X_i - median(H_d)|), X_i 属于 H_d

MADScore(X,w,d)
    = (X_d - median_past_d) / (1.4826 * MAD_past_d + 1e-12)
```

当前实现的规则是：

- 默认历史窗口为 10 条记录，最少需要 5 个有效历史样本。
- 当历史 `MAD <= 0` 时，异常分数直接设为空值。
- 正分表示当前值高于历史中位数，负分表示低于历史中位数。
- 分数的绝对值越大，表示相对近期常态的偏离越明显。


### 2.3 四个因子共用的价格趋势过滤

移动平均包含当天收盘价，并要求窗口内有完整的 `n` 个有效样本：

```text
MA(n,d) = mean(C_d, C_{d-1}, ..., C_{d-n+1})

ma5_ma10_bias_d  = MA(5,d) / MA(10,d) - 1
ma10_ma20_bias_d = MA(10,d) / MA(20,d) - 1
```

四个因子都要求以下公共趋势条件成立：

```text
TrendFilter_d =
       MA(5,d) > MA(10,d) > MA(20,d)
   and ma5_ma10_bias_d  >= 0.01
   and ma10_ma20_bias_d >= 0.01
   and MA(5,d) > MA(120,d)
```

其中代码把除 `MA(5,d) > MA(120,d)` 外的条件保存为 `ma_bull_stack_filter`，把这一长期趋势条件单独保存为 `ma5_gt_ma120_filter`。这意味着信号最早也要等 120 日均线形成后才可能触发。


## 3. 因子概览

| ID | 名称 | 识别状态 | `factor_value` |
| --- | --- | --- | --- |
| 11 | `price_up_volume_oi_surge` | 强势价格背景下，成交放大且 5 日持仓变化异常 | `price_volume_oi_score` |
| 12 | `price_up_speculation_up` | 价格与持仓均线偏强，同时投机度异常升高 | `speculation_mad_score` |
| 13 | `price_up_oi_down` | 价格趋势仍强，但近期高位持仓当日回落 | `log_open_interest_mad_score` |
| 14 | `uptrend_crowded_chase` | 上涨过程中持仓继续增加，成交或振幅异常 | `crowded_chase_score` |

## 4. 11 号：price_up_volume_oi_surge

### 4.1 识别目标

识别价格处于近期高位且继续上涨，同时成交量明显放大、5 日持仓变化率显著偏离近期常态的交易日。持仓异常取绝对值，因此既包括异常增仓，也包括异常减仓。

### 4.2 指标

价格指标：

```text
ret_5_d = C_d / C_{d-5} - 1

close_rank_20_d = Rank(C, 30, d)，最少 8 个历史样本
close_rank_60_d = Rank(C, 60, d)，最少 20 个历史样本
```

需要注意：字段名仍为 `close_rank_20`，但 11 号脚本的实际短分位窗口是 **30 日**。`close_rank_60` 只进入输出特征，不参与信号和连续得分。

成交量指标：

```text
log_volume_d      = log(V_d)，仅在 V_d > 0 时计算
volume_mad_score_d = MADScore(log_volume, 10, d)
```

持仓变化指标：

```text
oi_change_5_d
    = OI_d - OI_{d-5}

oi_change_5_rate_d
    = (OI_d - OI_{d-5}) / OI_{d-5}

oi_change_5_rate_mad_score_d
    = MADScore(oi_change_5_rate, 10, d)

oi_change_5_rate_mad_abs_score_d
    = |oi_change_5_rate_mad_score_d|
```

计算变化率时，若分母 `OI_{d-5}` 为 `0`，代码将分母视为空值。


### 4.3 信号条件

以下条件必须全部成立：

```text
signal_11_d = 1(
       TrendFilter_d
   and close_rank_20_d >= 0.90
   and ret_5_d > 0
   and oi_change_5_rate_mad_abs_score_d >= 1.0
   and volume_mad_score_d >= 1.0
)
```

这里的 `close_rank_20_d` 实际是过去 30 条记录的历史分位。持仓条件不限制方向，只要求 5 日持仓变化率的异常程度达标；成交量条件则只接受正向放量。

## 5. 12 号：price_up_speculation_up

### 5.1 识别目标

识别价格均线处于强势结构、短期持仓均线高于中期持仓均线，同时成交相对持仓的投机度异常升高的交易日。

### 5.2 指标

持仓均线包含当天持仓量，且分别要求完整的 5 个和 10 个有效样本：

```text
OI_MA(5,d)  = mean(OI_d, ..., OI_{d-4})
OI_MA(10,d) = mean(OI_d, ..., OI_{d-9})
```

投机度及其异常分数：

```text
speculation_d
    = log(V_d / OI_d)

speculation_mad_score_d
    = MADScore(speculation, 10, d)，最少 5 个历史样本
```

### 5.3 信号条件

```text
signal_12_d = 1(
       TrendFilter_d
   and OI_MA(5,d) > OI_MA(10,d)
   and speculation_mad_score_d >= 1.0
)
```

12 号没有额外要求价格处于历史高分位，也没有单独要求 5 日收益为正；价格部分完全由公共趋势过滤决定。

## 6. 13 号：price_up_oi_down

### 6.1 识别目标

识别价格仍处于强势趋势，但持仓量在近期相对高位上较前一日下降的交易日。它描述的是“高位持仓回落”，不等同于持仓量已经跌到低位。

### 6.2 指标

13 号脚本会再次把 `OI_d <= 0` 的值设为空值，然后计算：

```text
open_interest_rank_10_d
    = Rank(OI, 10, d)，最少 5 个历史样本

log_open_interest_d
    = log(OI_d)

delta_log_open_interest_d
    = log(OI_d) - log(OI_{d-1})

oi_change_rate_d
    = (exp(delta_log_open_interest_d) - 1) * 100
    = (OI_d / OI_{d-1} - 1) * 100

log_open_interest_mad_score_d
    = MADScore(log_open_interest, 10, d)
```

`oi_change_rate` 的单位是百分比，例如数值 `-2` 表示持仓量较前一日下降约 `2%`。


### 6.3 信号条件

```text
signal_13_d = 1(
       TrendFilter_d
   and open_interest_rank_10_d >= 0.80
   and delta_log_open_interest_d < 0
)
```

关键区别：

- `open_interest_rank_10 >= 0.80` 要求当前持仓量仍高于或等于过去窗口中至少 80% 的有效值。
- `delta_log_open_interest < 0` 要求持仓量当日确实下降。
- `log_open_interest_mad_score` 只作为 `factor_value` 输出，**不参与信号触发**。

13 号不看成交量、投机度、5 日收益或持仓 MAD 阈值。

## 7. 14 号：uptrend_crowded_chase

### 7.1 识别目标

识别价格已处于近期高位并继续上涨、最近 5 日持仓明显增加，同时成交量或日内振幅出现异常的交易日，用来刻画趋势中的继续追入与交易拥挤。

### 7.2 指标

价格与持仓变化：

```text
ret_5_d = C_d / C_{d-5} - 1

close_rank_20_d = Rank(C, 20, d)，最少 8 个历史样本
close_rank_60_d = Rank(C, 60, d)，最少 20 个历史样本

log_open_interest_d = log(OI_d)，仅在 OI_d > 0 时计算

oi_ret_5_d
    = log(OI_d) - log(OI_{d-5})
    = log(OI_d / OI_{d-5})
```

与 11 号不同，14 号的 `close_rank_20` 确实使用 20 日窗口。`close_rank_60` 只进入输出特征。

成交量、振幅和收盘位置：

```text
log_volume_d = log(V_d)，仅在 V_d > 0 时计算

range_pct_d
    = (H_d - L_d) / C_d

close_location_d
    = (C_d - L_d) / (H_d - L_d)

open_interest_mad_score_d
    = MADScore(log_open_interest, 10, d)

volume_mad_score_d
    = MADScore(log_volume, 10, d)

range_mad_score_d
    = MADScore(range_pct, 10, d)
```

当 `C_d = 0` 时，`range_pct` 为空；当 `H_d = L_d` 时，`close_location` 为空。这些空值不会通过相应信号条件。


### 7.3 信号条件

```text
signal_14_d = 1(
       TrendFilter_d
   and close_rank_20_d >= 0.90
   and ret_5_d > 0
   and oi_ret_5_d > 0.05
   and (
          volume_mad_score_d >= 1.2
       or range_mad_score_d  >= 1.2
   )
   and close_location_d <= 0.95
)
```

其中：

- `oi_ret_5 > 0.05` 是严格大于，等价于 5 日持仓增幅大于 `exp(0.05)-1`，约为 `5.13%`。
- 成交量异常和振幅异常满足任意一项即可。
- `close_location <= 0.95` 沿用当前代码，排除几乎收在当日最高点的 K 线。
- 信号使用 5 日持仓对数变化，连续得分使用持仓量水平的 MAD 分数，二者不能混用。

## 8. 计算时点、缺失值和输出

### 8.1 计算时点

价格均线、收益、当日成交量、当日持仓量、振幅和收盘位置都包含当天数据，因此信号是在当日数据完整后确认。历史分位和 MAD 的参照窗口不包含当天，也没有使用未来数据。

若把信号映射为实际交易，不能在没有额外成交假设的情况下直接使用当天收盘价成交；交易时点和收益评价由后续回测或事件研究单独定义。

### 8.2 缺失值

任一硬条件所需指标为空时，比较结果不会成立，因此当天 `signal = 0`。连续得分对空值的处理则因公式而异：

- 11、14 号通过 `fillna(0)` 或 `positive_part()` 将得分项空值按 `0` 处理。
- 12、13 号直接使用 MAD 分数，MAD 分数为空时 `factor_value` 也为空。

### 8.3 标准输出

`code/chapter1/run/run_all.py` 会依次准备日频数据、对全部品种运行 11—14 号因子，并更新 combined 图表。每个因子的默认结果目录包含：

```text
results/chapter1/{因子运行目录}/
  factors/{SYMBOL}_{factor_id}_{factor_name}.csv
  signals/{SYMBOL}_{factor_id}_{factor_name}_signals.csv
  summary/{SYMBOL}_{factor_id}_{factor_name}_summary.csv
  summary/all_symbols_{factor_id}_{factor_name}_summary.csv
```

其中：

- `factors/` 保存完整日频指标、`factor_value` 和 `signal`。
- `signals/` 只保存 `signal = 1` 的日期和特征。
- `summary/` 保存单品种及全品种汇总。

综合信号看板：

https://ziyu-ge.github.io/Volume-OI-Futures-Research/results/chapter1/combined/figures/combined_factor_signals_dashboard.html
