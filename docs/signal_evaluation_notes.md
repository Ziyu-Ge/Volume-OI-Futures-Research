# 信号测试逻辑说明

## 测试目的

本测试用于检查 11、12、13、14 号因子发出信号后，价格在未来 3、5、10 个交易日后是否出现足够大的收盘回落。

这里不是完整交易回测，不模拟开仓、平仓、手续费、滑点和资金曲线。它只做事件研究：

```text
因子信号出现后，观察窗口最后一天的收盘价，相对事件日收盘价是否回落到动态阈值以上？
```

如果达到阈值，就认为这次信号对后续回落风险有提示作用。

## 输入数据

测试脚本读取各因子已经生成的日频因子表：

```text
results/{factor_run_dir}/tables/factors/{SYMBOL}_{factor_id}_{factor_name}.csv
```

主要使用字段：

- `date`：交易日期。
- `close`：默认评估价格，可通过 `--price-column` 改成其他价格列。
- `signal`：因子信号，`1` 表示当天触发信号。
- `factor_id`、`factor_name`：用于分组汇总。

默认评估因子为 `11,12,13,14`，默认观察窗口为 `3,5,10` 个交易日。

## 动态阈值

设事件日为 `t`，收盘价为 `P_t`，日收益率为：

```text
r_t = P_t / P_{t-1} - 1
```

测试用事件日前 20 个交易日的平均绝对收益率衡量近期波动：

```text
vol_t = mean(|r_{t-i}|), i = 1, 2, ..., 20
```

动态回撤阈值为历史波动的 2 倍：

```text
theta_t = 2 * vol_t
```

代码中使用 `shift(1)`，所以阈值只使用事件日以前的数据，不使用当天或未来信息。若历史样本不足、阈值缺失或阈值非正，则该事件不参与胜率统计。

## 信号簇

因子可能连续多天触发信号。为避免同一段行情重复计票，测试会先把相近信号合并为信号簇：

```text
两个信号之间的无信号间隔 <= 10 个交易日，则归为同一个信号簇
```

每个信号簇只取最后一个信号日作为事件日：

```text
event_date = last_signal_date_in_cluster
```

这表示把一段连续或相近的拥挤信号看成一次事件，从该事件结束处开始观察后续价格变化。

## 胜负判断

对事件日 `t` 和观察窗口 `k`，取窗口最后一天的收盘价：

```text
P_{t+k}
```

新的胜负判断使用“事件日收盘价到观察窗口最后一天收盘价”的回撤比例：

```text
ED_{t,k} = max(P_t - P_{t+k}, 0) / P_t
```

其中 `ED` 是 end-date drawdown。因为动态阈值 `theta_t` 也是收益率比例，所以价格差要除以事件日收盘价后再比较。

如果观察窗口最后一天的收盘回撤达到动态阈值，则记为命中：

```text
win_{t,k} = 1(ED_{t,k} >= theta_t)
```

否则记为未命中：

```text
win_{t,k} = 0
```

若事件日后不足 `k` 个有效价格，则该事件标记为不可评价，不进入胜率分母。

## 最大回撤保留为观察指标

脚本仍然保留未来窗口内最大回撤，用来观察信号后价格曾经向下走到多深：

```text
DD_{t,k} = max(P_t - min(P_{t+1}, ..., P_{t+k}), 0) / P_t
```

但 `DD_{t,k}` 不再用于判断 `win`。现在的命中只看 `ED_{t,k}`，也就是观察窗口最后一天收盘价相对事件日收盘价的回撤。

因此可能出现：

```text
DD_{t,k} >= theta_t，但 ED_{t,k} < theta_t
```

这种情况表示窗口中间曾经跌到阈值以下，但到第 `k` 天收盘时已经反弹回来，所以不算命中。

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

基准胜率不看因子信号，而是把每个可评价交易日都当作普通事件日，用同样的末日收盘回撤规则计算：

```text
baseline_win_rate_k = baseline_wins_k / baseline_samples_k
```

信号相对基准的优势为：

```text
win_rate_diff_k = strategy_win_rate_k - baseline_win_rate_k
```

如果 `win_rate_diff_k > 0`，说明信号日后的末日收盘回撤概率高于普通交易日。

## 输出字段

事件明细保存在：

```text
results/evaluation/tables/events/signal_cluster_events.csv
```

主要字段：

- `signal_cluster_start_date`：信号簇开始日期。
- `signal_date`：信号簇最后一天，也是事件日。
- `signal_close`：事件日收盘价。
- `cluster_signal_days`：信号簇内的信号日数量。
- `threshold`：事件日动态阈值 `theta_t`。
- `strategy_observation_end_drawdown`：观察窗口最后一天收盘回撤 `ED_{t,k}`，用于判断胜负。
- `strategy_observation_max_drawdown`：观察窗口内最大回撤 `DD_{t,k}`，只用于观察。
- `strategy_win`：是否命中。
- `is_evaluable`：是否进入胜率统计。

汇总结果分三层：

- `overall_by_lookahead.csv`：全部品种、全部因子的总体结果。
- `factor_by_lookahead.csv`：按因子汇总。
- `symbol_factor_by_lookahead.csv`：按品种和因子汇总。

## 运行命令

完整流程：

```bash
python3 code/run_all.py
python3 code/backtest/evaluate_signal_win_rate.py \
  --runs-dir results \
  --output-dir results/evaluation \
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
