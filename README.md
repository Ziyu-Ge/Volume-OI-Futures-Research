# Volume-OI Futures Research

期货量价持仓因子研究仓库。项目从分钟级行情数据出发，聚合日频数据，构造价格、成交量、持仓量和投机度相关特征，并运行 11、12、13、14 号因子，输出因子表、信号表、汇总表、静态图和组合信号 dashboard。

## 项目结构

```text
.
├── code/
│   ├── 00_prepare_data.py
│   ├── config.py
│   ├── run_all.py
│   ├── run_11_price_up_volume_oi_surge_all.py
│   ├── run_12_price_up_speculation_up_all.py
│   ├── run_13_price_up_oi_down_all.py
│   ├── run_14_uptrend_crowded_chase_all.py
│   ├── plot_combined_signals.py
│   └── factors/
│       ├── 11price_up_volume_oi_surge.py
│       ├── 12price_up_speculation_up.py
│       ├── 13price_up_oi_down.py
│       ├── 14uptrend_crowded_chase.py
│       └── volume_price_factor_utils.py
├── data/
├── docs/
└── results/
```

`data/` 存放原始分钟数据，`results/` 存放运行结果。这两个目录通常不作为源码提交。

## 数据要求

原始数据文件放在 `data/{SYMBOL}.csv`，例如：

```text
data/JD.csv
data/CU.csv
```

CSV 至少需要包含以下字段：

```text
datetime, open, close, high, low, volume, total_turnover, open_interest
```

`00_prepare_data.py` 会按交易日聚合分钟数据，并生成日频字段：

```text
date, open, close, high, low, volume, total_turnover, open_interest, speculation
```

其中：

```text
speculation = log(volume / open_interest)
```

## 环境依赖

项目当前没有包管理文件，可直接使用本机 Python 环境运行脚本。常用依赖包括：

```bash
pip install pandas numpy matplotlib plotly
```

本机脚本默认使用 `python3`。

## 快速开始

在仓库根目录运行。

### 1. 准备单品种日频数据

单品种入口读取 `code/config.py` 中的 `SYMBOL`，当前默认是 `JD`。

```bash
python3 code/00_prepare_data.py
```

输出示例：

```text
results/tables/daily/JD_daily.csv
```

### 2. 运行单个因子

```bash
python3 code/factors/11price_up_volume_oi_surge.py
python3 code/factors/12price_up_speculation_up.py
python3 code/factors/13price_up_oi_down.py
python3 code/factors/14uptrend_crowded_chase.py
```

### 3. 批量运行指定因子

批量 runner 会扫描 `data/*.csv`，将文件名识别为品种代码，并在子进程中临时注入 `SYMBOL`。

```bash
python3 code/run_11_price_up_volume_oi_surge_all.py
python3 code/run_12_price_up_speculation_up_all.py
python3 code/run_13_price_up_oi_down_all.py
python3 code/run_14_uptrend_crowded_chase_all.py
```

只运行部分品种：

```bash
python3 code/run_11_price_up_volume_oi_surge_all.py --symbols JD,CU --keep-going
```

使用临时输出目录，避免覆盖已有 `results/`：

```bash
python3 code/run_11_price_up_volume_oi_surge_all.py --symbols JD --output-dir /tmp/factor11_jd
```

只基于已有结果重新汇总：

```bash
python3 code/run_11_price_up_volume_oi_surge_all.py --collect-only
```

### 4. 运行全部因子和组合图

```bash
python3 code/run_all.py
```

也可以单独生成组合信号图：

```bash
python3 code/plot_combined_signals.py --runs-dir results --output-dir results/combined --factor-ids 11,12,13,14
```

只画指定品种：

```bash
python3 code/plot_combined_signals.py --runs-dir results --output-dir results/combined --symbols JD
```

## 输出说明

单个因子的标准输出目录结构：

```text
{output_dir}/
  tables/
    daily/
    factors/
    signals/
    summary/
  figures/
    factors/
```

主要文件：

```text
tables/daily/{SYMBOL}_daily.csv
tables/factors/{SYMBOL}_{factor_id}_{factor_name}.csv
tables/signals/{SYMBOL}_{factor_id}_{factor_name}_signals.csv
tables/summary/{SYMBOL}_{factor_id}_{factor_name}_summary.csv
figures/factors/{SYMBOL}_{factor_id}_{factor_name}_signal_on_price.png
figures/factors/{SYMBOL}_{factor_id}_{factor_name}_factor_value.png
```

组合输出默认位于：

```text
results/combined/
```

包含跨因子统计表、单品种组合信号 PNG/HTML，以及 dashboard。

## 当前因子

| ID | 因子名称 | 核心含义 |
| --- | --- | --- |
| 11 | `price_up_volume_oi_surge` | 价格处于高位并上涨，成交量放大，持仓变化异常 |
| 12 | `price_up_speculation_up` | 价格趋势向上，持仓参与度上升，投机度异常升高 |
| 13 | `price_up_oi_down` | 价格趋势仍强，但高位持仓开始下降 |
| 14 | `uptrend_crowded_chase` | 上涨趋势中持仓继续增加，成交或振幅异常，刻画追涨拥挤 |

更详细的因子逻辑见 [docs/factor_logic_notes.md](docs/factor_logic_notes.md)。

## 配置与运行机制

- 单品种脚本通过 `from config import SYMBOL` 读取品种代码。
- 批量 runner 不需要手动修改 `code/config.py`，会在子进程中临时注入目标品种。
- `RESULTS_OUTPUT_DIR` 可控制输出根目录。
- 公共工具会优先读取 `RESULTS_OUTPUT_DIR/tables/daily/` 下的日频数据，找不到时回退到默认 `results/tables/`。
- Matplotlib 和工具缓存会写入项目内 `.matplotlib/` 和 `.cache/`，避免污染用户全局目录。

## 因子开发约定

新增或修改因子时，优先复用 `code/factors/volume_price_factor_utils.py` 中的公共函数：

```text
load_daily()
past_rank()
mad_score()
past_mad_score()
positive_part()
add_price_ma_features()
add_volume_price_features()
save_factor_outputs()
```

因子脚本文件名需要以数字 ID 开头，例如：

```text
15new_factor_name.py
```

每个因子建议明确生成：

- 连续因子值字段 `factor_value`
- 二值信号字段 `signal`
- 用于解释信号的关键中间变量 `feature_columns`

如果新增因子，通常还需要同步考虑：

- 新增或复用批量 runner
- 更新 `code/run_all.py`
- 如需进入组合图，更新 `plot_combined_signals.py` 的默认 factor id

## 时间约定

- 历史分位和 MAD 基准只使用今天以前的数据。
- 均线、成交量、持仓和投机度条件包含当日收盘后的数据，因此信号应视为当日收盘后确认。
- 因子脚本只生成信号，不在同一文件中生成下一交易日仓位或交易方向。
- 后续如果做回测或实盘映射，应在独立模块中使用类似 `trade_signal = signal.shift(1)` 的方式处理。

## 验证命令

项目目前没有测试框架。修改代码后建议先做轻量验证：

```bash
python3 -m py_compile code/*.py code/factors/*.py
python3 code/run_11_price_up_volume_oi_surge_all.py --symbols JD --output-dir /tmp/factor11_jd
```

如果改动涉及组合图：

```bash
python3 code/plot_combined_signals.py --runs-dir results --output-dir /tmp/combined_jd --symbols JD
```

如果改动影响公共逻辑，建议分别抽一个品种运行 11、12、13、14 号因子，并确认 `tables/summary/`、`tables/signals/` 和 `figures/factors/` 都能正常生成。
