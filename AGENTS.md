# AGENTS.md

## 项目概览

本项目是一个期货量价持仓因子研究仓库。核心流程是：

1. 从 `data/{SYMBOL}.csv` 读取分钟数据。
2. 用 `code/00_prepare_data.py` 聚合为日频数据，并计算投机度 `speculation = log(volume / open_interest)`。
3. 在 `code/factors/` 下运行 11、12、13、14 号因子脚本，生成每日因子表、信号表、汇总表和图。
4. 用 `code/plot_combined_signals.py` 汇总多个因子的信号，输出组合图表和 HTML dashboard。

项目当前没有包管理文件和测试框架配置，主要以可直接运行的 Python 脚本组织。常用依赖包括 `pandas`、`numpy`、`matplotlib`、`plotly`。

## 目录结构

- `code/config.py`：默认单品种配置，当前 `SYMBOL = "JD"`。
- `code/00_prepare_data.py`：分钟数据转日频数据，输出到 `results/tables/daily/` 或 `RESULTS_OUTPUT_DIR/tables/daily/`。
- `code/factors/`：因子实现脚本和公共工具。
- `code/factors/volume_price_factor_utils.py`：公共函数，包括读取日频、历史分位、MAD score、均线特征、量价特征、标准化保存输出。
- `code/run_11_*.py` 到 `code/run_14_*.py`：按品种批量运行单个因子，并生成 all-symbol summary。
- `code/run_all.py`：依次运行 11、12、13、14 号因子的批量脚本，再运行组合信号绘图。
- `code/plot_combined_signals.py`：从各因子输出目录收集信号，生成组合统计表、PNG、HTML 和 dashboard。
- `docs/`：预留研究说明文档目录；当前流程不依赖固定文档。
- `data/`：原始分钟 CSV 数据，已在 `.gitignore` 中忽略。
- `results/`：生成结果，已在 `.gitignore` 中忽略。

## 数据约定

`data/{SYMBOL}.csv` 至少应包含以下列：

- `datetime`
- `open`
- `close`
- `high`
- `low`
- `volume`
- `total_turnover`
- `open_interest`

品种代码来自 CSV 文件名，例如 `data/JD.csv` 对应 `JD`。批量 runner 会扫描 `data/*.csv` 并将文件名转为大写品种代码。

日频输出字段由 `00_prepare_data.py` 生成，包括：

- `date`
- `open`
- `close`
- `high`
- `low`
- `volume`
- `total_turnover`
- `open_interest`
- `speculation`

不要把 `data/`、`results/`、`.cache/`、`.matplotlib/` 或 `.DS_Store` 当成源代码改动提交。

## 常用命令

在仓库根目录运行命令。本机使用 `python3` 运行 Python 脚本。

单品种准备日频数据，使用 `code/config.py` 中的 `SYMBOL`：

```bash
python3 code/00_prepare_data.py
```

单品种运行某个因子，仍使用 `code/config.py` 中的 `SYMBOL`：

```bash
python3 code/factors/11price_up_volume_oi_surge.py
python3 code/factors/12price_up_speculation_up.py
python3 code/factors/13price_up_oi_down.py
python3 code/factors/14uptrend_crowded_chase.py
```

批量运行指定因子，默认会扫描 `data/` 下全部品种：

```bash
python3 code/run_11_price_up_volume_oi_surge_all.py
python3 code/run_12_price_up_speculation_up_all.py
python3 code/run_13_price_up_oi_down_all.py
python3 code/run_14_uptrend_crowded_chase_all.py
```

只跑部分品种：

```bash
python3 code/run_11_price_up_volume_oi_surge_all.py --symbols JD,CU --keep-going
```

指定输出目录，适合验证改动时避免覆盖已有 `results/`：

```bash
python3 code/run_11_price_up_volume_oi_surge_all.py --symbols JD --output-dir /tmp/factor11_jd
```

只基于已有结果重新汇总 summary：

```bash
python3 code/run_11_price_up_volume_oi_surge_all.py --collect-only
```

运行全部因子并生成组合图：

```bash
python3 code/run_all.py
```

单独生成组合信号图：

```bash
python3 code/plot_combined_signals.py --runs-dir results --output-dir results/combined --factor-ids 11,12,13,14
```

只画指定品种：

```bash
python3 code/plot_combined_signals.py --runs-dir results --output-dir results/combined --symbols JD
```

## 运行机制注意事项

- 单品种脚本通过 `from config import SYMBOL` 读取品种。直接运行因子脚本时，如需换品种，要改 `code/config.py` 或用批量 runner 的 `--symbols`。
- 批量 runner 会在子进程中临时注入 `config.SYMBOL`，因此不需要为了批量运行修改 `code/config.py`。
- `RESULTS_OUTPUT_DIR` 可控制输出根目录。公共函数会优先读取该目录下的日频数据；如果找不到，也会回退到默认 `results/tables/`。
- Matplotlib 和工具缓存被设置到项目内 `.matplotlib/`、`.cache/`，避免写入用户全局目录。

## 输出约定

单个因子的标准输出目录结构如下：

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

因子脚本通过 `save_factor_outputs()` 统一生成：

- 每日因子表：`tables/factors/{SYMBOL}_{factor_id}_{factor_name}.csv`
- 信号事件表：`tables/signals/{SYMBOL}_{factor_id}_{factor_name}_signals.csv`
- 汇总表：`tables/summary/{SYMBOL}_{factor_id}_{factor_name}_summary.csv`
- 价格信号图：`figures/factors/{SYMBOL}_{factor_id}_{factor_name}_signal_on_price.png`
- 因子值图：`figures/factors/{SYMBOL}_{factor_id}_{factor_name}_factor_value.png`

组合输出默认在 `results/combined/` 下，包含跨因子的统计表、每个品种的组合信号 PNG/HTML，以及 dashboard。

## 因子实现约定

- 因子脚本文件名必须以数字 ID 开头，例如 `11price_up_volume_oi_surge.py`。`parse_factor_script_metadata(__file__)` 会从文件名解析 `factor_id` 和 `factor_name`。
- 新因子优先复用 `volume_price_factor_utils.py` 中的公共函数：
  - `load_daily()`
  - `past_rank()`
  - `mad_score()` / `past_mad_score()`
  - `positive_part()`
  - `add_price_ma_features()`
  - `add_volume_price_features()`
  - `save_factor_outputs()`
- 每个因子应明确生成一个连续 `factor_value` 字段和一个二值 `signal` 字段，再交给 `save_factor_outputs()`。
- `feature_columns` 应包含解释信号所需的关键中间变量，方便后续从因子表、信号表和 summary 回溯。
- 如果新增因子，需要同步考虑：
  - 新增 `code/factors/{id}{name}.py`
  - 新增或复用批量 runner
  - 更新 `code/run_all.py`
  - 如需进入组合图，更新 `plot_combined_signals.py` 的默认 factor id

## 时间和未来函数约定

- 历史分位和 MAD 基准应只使用今天以前的数据。现有 `past_rank()`、`mad_score()` 和 `past_mad_score()` 已按这个原则实现。
- 均线、成交量、持仓和投机度条件可以包含当日收盘后的数据，因此信号只能视为当日收盘后确认。
- 因子文件只负责生成信号，不应在同一文件里直接生成下一交易日仓位或交易方向。
- 如果后续做回测或实盘映射，应在独立模块中使用类似 `trade_signal = signal.shift(1)` 的方式处理。

## 修改建议

- 改因子逻辑前，先读对应 `code/factors/*.py`；如涉及公共特征或输出逻辑，也要读 `code/factors/volume_price_factor_utils.py` 和相关 runner。
- 改公共函数前，检查 11、12、13、14 号因子是否都依赖该函数，避免改变已有输出字段含义。
- 保持输出文件名和列名稳定；已有绘图与汇总逻辑依赖这些命名。
- 大规模重跑可能耗时且会改写 `results/`。验证代码时优先使用 `--symbols` 和临时 `--output-dir`。
- 不要为了单次验证把生成结果提交为源码改动。

## 验证建议

项目没有现成自动化测试。完成代码改动后，优先做轻量验证：

```bash
python3 -m py_compile code/*.py code/factors/*.py
python3 code/run_11_price_up_volume_oi_surge_all.py --symbols JD --output-dir /tmp/factor11_jd
```

如果改动涉及组合图，再运行：

```bash
python3 code/plot_combined_signals.py --runs-dir results --output-dir /tmp/combined_jd --symbols JD
```

如果改动会影响所有因子的公共逻辑，至少分别抽一个品种跑 11、12、13、14 号因子，确认 `tables/summary/`、`tables/signals/` 和 `figures/factors/` 都能生成。
