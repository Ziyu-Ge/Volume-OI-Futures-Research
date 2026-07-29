# Volume-OI Futures Research

期货量价持仓因子研究仓库。项目以 `data/` 下的分钟行情 CSV 为输入，围绕价格、成交量、持仓量和投机度构建信号，并按研究章节组织为三条主线：

- `chapter1`：日频量价持仓信号因子 11-14，生成因子表、信号表、汇总表、组合图和信号胜率评估。
- `chapter2`：高乖离持仓回落类空头因子 21-24，支持日频开平仓、日频开仓小时级平仓，以及 24 号因子的滚动样本外参数选择。
- `chapter3`：龙头与跟随品种识别，以及基于跟随信号的做空回测、参数网格和可视化复盘。

## 项目结构

```text
.
├── code/
│   ├── chapter1/
│   │   ├── 00_prepare_data.py
│   │   ├── factors/
│   │   ├── run/run_all.py
│   │   ├── plot/
│   │   └── backtest/
│   ├── chapter2/
│   │   ├── prepare_data.py
│   │   ├── run_factor.py
│   │   ├── factors/
│   │   ├── rules/
│   │   ├── engines/
│   │   ├── rolling/
│   │   └── core/
│   └── chapter3/
│       ├── common/
│       ├── run/
│       ├── backtest/
│       ├── factors/
│       └── plot/
├── data/
├── docs/
├── useful_plots/
└── results/
```

`data/` 存放原始分钟数据，`results/` 存放运行结果。这两个目录通常包含大文件或批量输出，不建议作为源码的一部分提交。

## 数据要求

原始数据文件放在：

```text
data/{SYMBOL}.csv
```

例如：

```text
data/CU.csv
data/JD.csv
```

CSV 至少需要包含：

```text
datetime, open, close, high, low, volume, total_turnover, open_interest
```

其中 `chapter3` 的识别主流程最低只使用：

```text
datetime, open, high, low, close, open_interest
```

脚本会将 `open_interest <= 0` 视为无效，并在日频/小时频缓存中计算：

```text
speculation = log(volume / open_interest)
```

## 环境依赖

项目当前没有固定的包管理文件，可直接使用本机 Python 环境运行。常用依赖：

```bash
pip install pandas numpy matplotlib plotly
```

所有命令默认在仓库根目录执行。也可以用当前解释器显式运行：

```bash
python3 ...
```

## 快速开始

### Chapter 1：运行 11-14 号日频因子

一键准备日频缓存、运行全部 11-14 号因子，并生成组合信号图：

```bash
python3 code/chapter1/run/run_all.py
```

等价流程包括：

```text
code/chapter1/00_prepare_data.py
code/chapter1/factors/11price_up_volume_oi_surge.py
code/chapter1/factors/12price_up_speculation_up.py
code/chapter1/factors/13price_up_oi_down.py
code/chapter1/factors/14uptrend_crowded_chase.py
code/chapter1/plot/plot_combined_signals.py
```

单独准备 chapter1 日频缓存：

```bash
python3 code/chapter1/00_prepare_data.py
```

运行信号胜率评估：

```bash
python3 code/chapter1/backtest/evaluate_signal_win_rate.py
```

常用参数示例：

```bash
python3 code/chapter1/backtest/evaluate_signal_win_rate.py \
  --factor-ids 11,12,13,14 \
  --lookahead-days-list 3,5,10 \
  --cluster-max-gap 10
```

### Chapter 2：运行 21-24 号空头因子

先生成日频缓存：

```bash
python3 code/chapter2/prepare_data.py --frequency daily
```

运行需要小时级平仓的因子前，再生成小时频缓存：

```bash
python3 code/chapter2/prepare_data.py --frequency hourly
```

运行单个因子：

```bash
python3 code/chapter2/run_factor.py --factor 21
python3 code/chapter2/run_factor.py --factor 22
python3 code/chapter2/run_factor.py --factor 23
python3 code/chapter2/run_factor.py --factor 24
```

运行 24 号因子的滚动样本外版本：

```bash
python3 code/chapter2/run_factor.py --factor 24_rolling
```

指定输入和输出目录：

```bash
python3 code/chapter2/run_factor.py \
  --factor 24 \
  --daily-dir results/chapter2/tables/daily \
  --hourly-dir results/chapter2/tables/hourly \
  --output-dir /tmp/factor_24
```

### Chapter 3：识别龙头和跟随品种

运行小时级龙头/跟随识别：

```bash
python3 code/chapter3/run/run_identification.py
```

限制日期或品种：

```bash
python3 code/chapter3/run/run_identification.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --symbols CU,AL,ZN
```

基于“向下跟随信号”做空回测：

```bash
python3 code/chapter3/backtest/run_short_backtest.py --fee-rate 0.0001
```

运行三参数网格：

```bash
python3 code/chapter3/backtest/run_short_grid.py --fee-rates 0,0.0001
```

生成网络图和事件复盘页：

```bash
python3 code/chapter3/plot/visualize.py
```

chapter3 还包含两个日频跟随策略研究入口：

```bash
python3 code/chapter3/factors/factor31.py
python3 code/chapter3/factors/factor32.py
python3 code/chapter3/backtest/tune_factor32.py
```

## 输出目录

### Chapter 1

日频缓存：

```text
results/chapter1/tables/{SYMBOL}_daily.csv
```

因子输出：

```text
results/chapter1/{factor_run_dir}/
  factors/
  signals/
  summary/
```

组合图和 dashboard：

```text
results/chapter1/combined/
```

信号胜率评估：

```text
results/chapter1/evaluation/
```

### Chapter 2

缓存数据：

```text
results/chapter2/tables/daily/{SYMBOL}_daily.csv
results/chapter2/tables/hourly/{SYMBOL}_hourly.csv
```

标准因子输出：

```text
results/chapter2/factor_{ID}/
  tables/
    symbol_metrics.csv
    trades.csv
    curves.csv
    portfolio_curve.csv
  figures/
    signals/
    strategy/
    return_summary.png
    all_symbols_summary.png
```

滚动样本外输出：

```text
results/chapter2/factor_24_rolling/
  tables/selected_params.csv
  tables/symbol_metrics.csv
  tables/trades.csv
  tables/curves.csv
  tables/portfolio_curve.csv
  figures/
```

### Chapter 3

识别结果：

```text
results/chapter3/identification/
  leader_results.csv
  follower_results.csv
  daily_bars.csv
```

做空回测：

```text
results/chapter3/short_backtest/
  trades.csv
  daily_returns.csv
  metrics.csv
```

参数网格：

```text
results/chapter3/short_grid/
```

事件可视化：

```text
results/chapter3/figures/
  leader_follower_network.png
  event_review.html
```

factor31/factor32 输出：

```text
results/chapter3/factor31/
results/chapter3/factor32/
results/chapter3/factor32_tuning/
```

## 当前因子

### Chapter 1

| ID | 因子名称 | 核心含义 |
| --- | --- | --- |
| 11 | `price_up_volume_oi_surge` | 价格处于高位并上涨，成交量放大，持仓变化异常 |
| 12 | `price_up_speculation_up` | 价格趋势向上，持仓参与度上升，投机度异常升高 |
| 13 | `price_up_oi_down` | 价格趋势仍强，但高位持仓开始下降 |
| 14 | `uptrend_crowded_chase` | 上涨趋势中持仓继续增加，成交或振幅异常，刻画追涨拥挤 |

### Chapter 2

| ID | 因子名称 | 执行频率 | 核心含义 |
| --- | --- | --- | --- |
| 21 | `high_bias_oi_drop` | 日频 | 高乖离状态下，持仓和价格同时转弱后开空 |
| 22 | `high_bias_oi_drop_mixed` | 日频开仓、小时级平仓 | 21 号的小时级退出版本 |
| 23 | `high_bias_oi_speculation_drop` | 日频 | 在 21 号基础上增加投机度回落过滤 |
| 24 | `high_bias_oi_speculation_drop_mixed` | 日频开仓、小时级平仓 | 23 号的小时级退出版本 |
| 24_rolling | `factor_24_rolling` | 样本外滚动 | 对 24 号做 walk-forward 参数选择 |

### Chapter 3

| 模块 | 作用 |
| --- | --- |
| `run_identification.py` | 逐小时识别同板块龙头和跟随品种 |
| `run_short_backtest.py` | 龙头向下时，做空同板块跟随品种 |
| `run_short_grid.py` | 搜索收益阈值、持仓阈值、相关性阈值 |
| `factor31.py` | 用 chapter2 的 23 号信号识别龙头，做空相关跟随品种 |
| `factor32.py` | 23 号式信号 + 高关系度过滤的日频跟随策略 |
| `tune_factor32.py` | 对 factor32 做参数粗调 |

## 文档

更详细的研究说明见：

```text
docs/chapter1_factor_logic_notes.md
docs/chapter1_signal_evaluation_notes.md
docs/chapter2_logic_notes.md
docs/chapter2_factor_24_logic_notes.md
docs/chapter3_notes.md
```

`useful_plots/` 存放已筛选出的代表性图表，可用于写报告或快速查看阶段性结论。

## 运行约定

- 所有命令建议从仓库根目录运行。
- chapter1 的日频缓存默认在 `results/chapter1/tables/`。
- chapter2 的日频和小时频缓存默认在 `results/chapter2/tables/daily/` 和 `results/chapter2/tables/hourly/`。
- chapter2 的夜盘数据会归入下一个真实日盘交易日；如果尾部找不到下一个交易日，会用下一个工作日兜底。
- chapter3 的板块映射集中在 `code/chapter3/common/config.py`；未分类品种可以生成中间数据，但不会参与同板块跟随识别。
- Matplotlib 和工具缓存会写入项目内缓存目录，避免污染用户全局目录。

## 开发约定

新增或修改因子时，优先复用现有公共模块：

```text
code/chapter1/factors/volume_price_factor_utils.py
code/chapter2/rules/
code/chapter2/engines/
code/chapter2/core/
code/chapter3/common/
```

chapter1 新增因子通常需要：

- 在 `code/chapter1/factors/` 下新增以数字 ID 开头的脚本。
- 输出 `factor_value` 和 `signal`。
- 同步更新 `code/chapter1/run/run_all.py`。
- 如需进入组合图，同步更新 `code/chapter1/plot/` 的默认因子配置。

chapter2 新增因子通常需要：

- 在 `code/chapter2/factors/` 下新增 `factor_{ID}.py`。
- 明确 `FACTOR_ID`、`FACTOR_NAME`、`ENGINE`、`USE_SPECULATION`、`ENTRY_CONFIG` 和 `EXIT_CONFIG`。
- 如果新增 engine 或输出口径，优先扩展 `engines/` 和 `core/reports.py`，不要在因子文件里重复写回测流程。

chapter3 新增识别规则或板块配置时，优先放在：

```text
code/chapter3/common/config.py
code/chapter3/common/rules.py
```

策略回测、网格和可视化分别放在 `backtest/`、`factors/` 和 `plot/` 下，避免把研究入口和公共规则混在一起。

## 时间约定

- 历史分位、均值、MAD、相关性和回归基准只使用信号日以前的数据，避免未来函数。
- chapter1 因子信号应视为当日收盘后确认。
- chapter2 日频因子通常在信号日收盘后确认，下一交易日开盘执行；小时级退出因子使用小时 bar 判断平仓。
- chapter3 小时识别只使用当前小时已经可见的数据；做空回测在信号出现后的下一根分钟 K 线开盘入场，当日最后一根分钟 K 线收盘离场。
