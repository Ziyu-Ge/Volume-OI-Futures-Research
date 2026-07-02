# 信号胜率评估说明

本文档对应当前三套评估脚本：

```text
code/backtest/evaluate_signal_win_rate.py
code/backtest/evaluate_signal_win_rate_cluster_last_day.py
code/backtest/evaluate_signal_win_rate_low_medium_high.py
```

当前结果来自三个 summary 目录：

```text
results/evaluation/tables/summary
results/evaluation_cluster_last_day/tables/summary
results/evaluation_low_medium_high/tables/summary
```

## 口径概览

这组脚本不是完整交易回测，不模拟开仓、平仓、手续费、滑点、保证金、资金曲线或止盈止损。它做的是事件研究：

```text
因子信号出现并合并成信号簇后，
取信号簇最后一天作为事件日，
观察未来 3、5、10 个交易日后收盘价是否相对事件日收盘价回落到动态阈值以上。
```

若未来窗口最后一天的收盘回撤达到阈值，则记为命中。这个口径评估的是信号对“后续数日收盘回落风险”的提示能力。

三套脚本共用同一套核心评估函数，差别主要在输出视角：

| 脚本 | 输出目录 | 输出重点 |
| --- | --- | --- |
| `evaluate_signal_win_rate.py` | `results/evaluation` | 完整输出，包括总体、分因子、总体置信度、分因子置信度 |
| `evaluate_signal_win_rate_cluster_last_day.py` | `results/evaluation_cluster_last_day` | 只输出信号簇最后一天口径的总体和分因子汇总 |
| `evaluate_signal_win_rate_low_medium_high.py` | `results/evaluation_low_medium_high` | 只输出 `low`、`medium`、`high` 三档置信度汇总 |

当前 CSV 中，`evaluation_cluster_last_day` 的 `overall_by_lookahead.csv`、`factor_by_lookahead.csv` 与 `evaluation` 中同名表完全一致；`evaluation_low_medium_high` 的两张 summary 与 `evaluation` 的置信度分层表完全一致，只是文件名更聚焦。

## 输入数据

脚本从各因子运行目录读取日频因子表：

```text
results/{factor_run_dir}/tables/factors/{SYMBOL}_{factor_id}_{factor_name}.csv
```

默认评估：

- 因子：`11,12,13,14`
- 观察窗口：`3,5,10` 个交易日
- 评估价格：`close`
- 波动率窗口：20 个历史交易日
- 信号簇最大无信号间隔：10 个交易日
- 置信度回看窗口：10 个交易日

主要字段：

- `date`：交易日期。
- `close`：默认评估价格，可用 `--price-column` 改成其他价格列。
- `signal`：因子信号，`1` 表示当天触发。
- `factor_id`、`factor_name`：因子标识。
- `confidence_level`、`confidence_score`、`confidence_signal_days`、`confidence_recent_dates`：若输入或 signals 表已有会读取，但当前脚本会优先基于同一品种近 10 个交易日的信号密度重新计算。

## 动态阈值

设事件日为 `t`，评估价格为 `P_t`，日收益率为：

```text
r_t = P_t / P_{t-1} - 1
```

脚本用事件日前 20 个交易日的平均绝对收益率衡量近期波动：

```text
vol_t = mean(|r_{t-i}|), i = 1, 2, ..., 20
```

动态回撤阈值固定为近期波动的 1 倍：

```text
theta_t = 1.0 * vol_t
```

代码中使用 `shift(1)`，所以阈值只使用事件日以前的数据，不使用事件日当天或未来信息。若历史样本不足、阈值缺失或阈值非正，该事件不进入胜率分母。

当前总体样本的平均阈值约为 `1.23%`。

## 信号簇

因子可能连续多天触发信号。为避免同一段行情重复计票，脚本先把相近信号合并为信号簇：

```text
两个信号之间的无信号间隔 <= 10 个交易日，则归为同一个信号簇。
```

每个信号簇只取最后一个信号日作为事件日：

```text
event_date = last_signal_date_in_cluster
```

这表示把连续或相近的拥挤信号视为一次事件，从信号簇结束处开始观察后续价格表现。若运行时设置 `--cluster-max-gap -1`，每个原始信号会单独计票。

## 置信度分层

脚本按同一品种、所有被选中因子，在近 10 个交易日内出现过信号的交易日数量计算置信度。默认回看窗口由 `--confidence-window-days` 控制，当前默认值为 `10`。

设近 10 个交易日内有 `signal_day_count` 个交易日出现过至少一个信号，则：

```text
confidence_score = min(signal_day_count / 4, 1.0)
```

分层规则：

- `none`：0 个信号日。
- `low`：1 个信号日。
- `medium`：2-3 个信号日。
- `high`：4 个及以上信号日。

这里的 `high` 更准确地说是“近期信号密集度高”，不是模型主观确信度更高。当前结果里 `high` 组的平均阈值也更高，说明它更多出现在波动较大的环境中。

另外，summary 中置信度分层表的基准胜率不是按置信度条件重新筛出的基准，而是同一个无条件基准胜率复制到各置信度组中。

## 胜负判断

对事件日 `t` 和观察窗口 `k`，脚本取窗口最后一天的收盘价：

```text
P_{t+k}
```

胜负判断使用事件日收盘价到观察窗口最后一天收盘价的回撤比例：

```text
ED_{t,k} = max(P_t - P_{t+k}, 0) / P_t
```

若末日收盘回撤达到动态阈值，则记为命中：

```text
win_{t,k} = 1(ED_{t,k} >= theta_t)
```

若事件日后不足 `k` 个有效价格，或事件日价格、未来末日价格、动态阈值不可用，则该事件标记为不可评价，不进入胜率分母。

脚本也保留未来窗口内最大回撤：

```text
DD_{t,k} = max(P_t - min(P_{t+1}, ..., P_{t+k}), 0) / P_t
```

但 `DD_{t,k}` 不用于判断 `win`。当前命中只看 `ED_{t,k}`。因此如果窗口中间跌破阈值，但第 `k` 天收盘已经反弹回来，仍然不算命中。

## 胜率和基准

对同一个观察窗口 `k`，信号胜率为：

```text
strategy_win_rate_k = strategy_wins_k / strategy_samples_k
```

其中：

```text
strategy_wins_k = sum(win_{t,k})
strategy_samples_k = 可评价的信号簇事件数量
```

基准胜率不看因子信号，而是把同一批日频面板里的每个可评价交易日都当作普通事件日，用同样的末日收盘回撤规则计算：

```text
baseline_win_rate_k = baseline_wins_k / baseline_samples_k
```

信号相对基准的优势为：

```text
win_rate_diff_k = strategy_win_rate_k - baseline_win_rate_k
```

若 `win_rate_diff_k > 0`，说明信号簇事件日后的末日收盘回撤概率高于普通交易日。

## 当前总体结果

`overall_by_lookahead.csv` 显示，当前 4 个因子合并后，3、5、10 日窗口均显著高于基准胜率。

| 观察窗口 | 信号簇事件 | 可评价 / 命中 | 平均阈值 | 策略胜率 | 基准胜率 | 胜率差 | 最大观察回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 日 | 2406 | 2401 / 1099 | 1.23% | 45.77% | 31.29% | +14.49 pct | 29.52% |
| 5 日 | 2406 | 2399 / 1180 | 1.23% | 49.19% | 35.34% | +13.85 pct | 29.52% |
| 10 日 | 2406 | 2388 / 1287 | 1.23% | 53.89% | 39.53% | +14.36 pct | 42.34% |

解读：

- 三个窗口的胜率差集中在 `+13.85` 到 `+14.49` 个百分点，信号对未来收盘回落有稳定正贡献。
- 10 日窗口的绝对胜率最高，为 `53.89%`；3 日窗口的相对基准优势略高，为 `+14.49` 个百分点。
- 这更像“后续数日回落风险提示”信号，不是一个可以直接对应当天或隔日交易执行的系统。

## 当前分因子结果

`factor_by_lookahead.csv` 显示，4 个因子在三个窗口下都跑赢各自基准。其中 11、14 号因子的胜率差更高，12、13 号因子也为正但相对弱一些。

| 因子 | 含义 | 3 日胜率 / 差 | 5 日胜率 / 差 | 10 日胜率 / 差 |
| --- | --- | ---: | ---: | ---: |
| 11 | `price_up_volume_oi_surge` | 50.85% / +19.56 pct | 53.06% / +17.72 pct | 57.51% / +17.98 pct |
| 12 | `price_up_speculation_up` | 41.53% / +10.24 pct | 46.87% / +11.53 pct | 51.76% / +12.23 pct |
| 13 | `price_up_oi_down` | 41.84% / +10.55 pct | 44.73% / +9.39 pct | 49.67% / +10.14 pct |
| 14 | `uptrend_crowded_chase` | 49.91% / +18.62 pct | 52.88% / +17.53 pct | 57.38% / +17.85 pct |

研究上可以优先关注：

- 11 号因子：价格上涨、成交量和持仓同步放大的拥挤信号，三个窗口均表现最强或接近最强。
- 14 号因子：上涨趋势中的追涨拥挤信号，表现与 11 号因子接近。
- 12、13 号因子：整体仍然有效，但需要结合品种、期限和样本量做二次筛选。

## 当前置信度分层结果

`overall_by_confidence.csv` 和 `overall_by_low_medium_high.csv` 显示，置信度分层没有呈现“越高越好”的单调关系。当前样本中，`low` 和 `medium` 的胜率差通常高于 `high`。

| 置信度 | 3 日样本 / 胜率 / 差 | 5 日样本 / 胜率 / 差 | 10 日样本 / 胜率 / 差 |
| --- | ---: | ---: | ---: |
| low | 683 / 47.14% / +15.86 pct | 682 / 52.20% / +16.86 pct | 674 / 55.04% / +15.51 pct |
| medium | 988 / 45.95% / +14.66 pct | 988 / 50.30% / +14.96 pct | 985 / 55.74% / +16.21 pct |
| high | 730 / 44.25% / +12.96 pct | 729 / 44.86% / +9.52 pct | 729 / 50.34% / +10.81 pct |

10 日窗口下，分因子、分置信度的主要结果如下：

| 因子 | low 胜率 / 差 | medium 胜率 / 差 | high 胜率 / 差 |
| --- | ---: | ---: | ---: |
| 11 | 60.00% / +20.47 pct | 60.26% / +20.73 pct | 50.90% / +11.37 pct |
| 12 | 51.59% / +12.06 pct | 55.09% / +15.56 pct | 48.05% / +8.52 pct |
| 13 | 49.69% / +10.16 pct | 49.25% / +9.72 pct | 50.27% / +10.74 pct |
| 14 | 57.93% / +18.40 pct | 59.56% / +20.03 pct | 53.42% / +13.89 pct |

结论：

- `high` 不应直接理解为更高预测把握度。它代表近期信号更密集，可能对应行情更拥挤、波动更高或已进入尾段。
- 11、14 号因子在 `low`、`medium` 档表现尤其强，10 日胜率接近或超过 58%-60%。
- 13 号因子分层差异较小，整体贡献更稳定但优势不如 11、14 明显。

## 品种-因子明细

当前 summary 目录不输出品种-因子层面的 CSV，只保留总体和因子层面的轻量汇总。若需要按品种和因子筛选，可以从事件表重新聚合：

```text
results/evaluation/tables/events/signal_cluster_events.csv
results/evaluation_cluster_last_day/tables/events/signal_cluster_last_day_events.csv
results/evaluation_low_medium_high/tables/events/signal_cluster_events_low_medium_high.csv
```

可按以下字段重新聚合：

```text
symbol, factor_id, factor_name, lookahead_days, confidence_level
```

事件表保留了每个信号簇事件的完整明细，因此可以继续做品种筛选、最小样本数过滤、跨窗口稳定性检查和分置信度复核。

## 输出字段

事件明细主要字段：

- `signal_cluster_start_date`：信号簇开始日期。
- `signal_date`：信号簇最后一天，也是事件日。
- `signal_close`：事件日收盘价。
- `cluster_signal_days`：信号簇内的信号日数量。
- `confidence_level`：事件日置信度分层。
- `confidence_score`：事件日置信度分数。
- `confidence_signal_days`：置信度回看窗口内出现过信号的交易日数量。
- `confidence_recent_dates`：置信度回看窗口内的信号日期。
- `threshold`：事件日动态阈值 `theta_t`。
- `strategy_observation_end_drawdown`：观察窗口最后一天收盘回撤 `ED_{t,k}`，用于判断胜负。
- `strategy_observation_max_drawdown`：观察窗口内最大回撤 `DD_{t,k}`，只用于观察。
- `strategy_win`：是否命中。
- `is_evaluable`：是否进入胜率统计。

summary 表统一保留以下字段：

- `factor_id`：因子 ID；总体汇总为 `ALL_FACTORS`。
- `factor_name`：因子名称；总体汇总为 `all_factors`。
- `lookahead_days`：观察窗口。
- `confidence_level`：置信度分层；非置信度分层表为 `ALL_CONFIDENCE`。
- `threshold`：分组内可评价事件的平均动态阈值。
- `strategy_win_rate`：信号胜率。
- `baseline_win_rate`：基准胜率。
- `win_rate_diff`：信号胜率减基准胜率。
- `strategy_observation_max_drawdown`：分组内观察窗口最大回撤的最大值。

`results/evaluation/tables/summary` 输出四张表：

- `overall_by_lookahead.csv`：全部品种、全部因子的总体结果。
- `factor_by_lookahead.csv`：按因子汇总。
- `overall_by_confidence.csv`：全部品种、全部因子，按置信度分层汇总。
- `factor_by_confidence.csv`：按因子和置信度分层汇总。

两个轻量输出目录对应：

- `results/evaluation_cluster_last_day/tables/summary/overall_by_lookahead.csv`
- `results/evaluation_cluster_last_day/tables/summary/factor_by_lookahead.csv`
- `results/evaluation_low_medium_high/tables/summary/overall_by_low_medium_high.csv`
- `results/evaluation_low_medium_high/tables/summary/factor_by_low_medium_high.csv`

## 图形输出

若不设置 `--skip-plots`，脚本会输出每个品种、因子、观察窗口对应的价格信号图：

```text
{output_dir}/figures/signal_clusters/{lookahead_days}d/
```

图中保留价格线和三类事件点：

- 黑色线：评估价格，默认是 `close`。
- 绿色上三角：可评价且命中的信号簇事件。
- 红色叉号：可评价但未命中的信号簇事件。
- 灰色圆点：不可评价的信号簇事件。

图中不绘制原始信号点和信号簇区间阴影，便于快速查看事件日和胜负结果。

## 运行命令

完整输出：

```bash
python3 code/backtest/evaluate_signal_win_rate.py \
  --runs-dir results \
  --output-dir results/evaluation \
  --factor-ids 11,12,13,14
```

只输出信号簇最后一天口径的总体和分因子结果：

```bash
python3 code/backtest/evaluate_signal_win_rate_cluster_last_day.py \
  --runs-dir results \
  --output-dir results/evaluation_cluster_last_day \
  --factor-ids 11,12,13,14
```

只输出 `low`、`medium`、`high` 分层结果：

```bash
python3 code/backtest/evaluate_signal_win_rate_low_medium_high.py \
  --runs-dir results \
  --output-dir results/evaluation_low_medium_high \
  --factor-ids 11,12,13,14
```

只评估单个品种：

```bash
python3 code/backtest/evaluate_signal_win_rate.py \
  --runs-dir results \
  --output-dir /tmp/evaluation_jd \
  --symbols JD
```

只评估 10 日窗口：

```bash
python3 code/backtest/evaluate_signal_win_rate.py \
  --runs-dir results \
  --output-dir /tmp/evaluation_10d \
  --lookahead-days 10
```

只输出表格、不生成信号图：

```bash
python3 code/backtest/evaluate_signal_win_rate.py \
  --runs-dir results \
  --output-dir results/evaluation \
  --skip-plots
```
