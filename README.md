# Volume-OI Futures Research

## 1. 项目结构

```text
.
├── .gitignore
├── README.md
├── code/
│   ├── chapter1/
│   │   ├── 00_prepare_data.py
│   │   ├── backtest/                  # 信号胜率事件研究
│   │   ├── factors/                   # 11—14 号量价持仓因子
│   │   ├── plot/                      # 多因子静态图和交互图
│   │   └── run/                       # Chapter 1 运行代码
│   ├── chapter2/
│   │   ├── core/                      # 指标、I/O、绩效、路径、绘图和报告
│   │   ├── engines/                   # 日频及小时级回测
│   │   ├── factors/                   # 21—24 号因子参数
│   │   ├── rolling/                   # 24 号因子滚动样本外选参
│   │   ├── rules/                     # 开仓和平仓规则
│   │   ├── prepare_data.py            # 数据处理
│   │   └── run_factor.py              # 运行代码
│   └── chapter3/
│       ├── backtest/                  # 跟随策略回测、网格和调参
│       ├── common/                    # 配置、行情处理和识别规则
│       ├── factors/                   # 31、32 号龙头—跟随因子
│       ├── plot/                      # 净值、价格路径和事件复盘图
│       └── run/                       # 龙头—跟随识别运行代码
├── data/                              # data/<SYMBOL>.csv 原始分钟行情
│   └── LC.csv                         # Git 中保留的 LC 数据文件
├── docs/                              # 三个章节的研究笔记
├── useful_plots/                      # 已筛选的代表性图表和离线网页
└── results/                           # 运行时生成，已被 Git 忽略
    ├── chapter1/
    ├── chapter2/
    └── chapter3/
```

`code/`、`data/LC.csv`、`docs/` 和 `useful_plots/` 是版本库内容；其他 `data/*.csv`、整个 `results/`、Python/Matplotlib/Pytest 缓存及 `.DS_Store` 都是本地输入、运行产物或工具缓存，不属于受版本控制的源码。

## 2. 如何运行

所有命令都在仓库根目录执行。仓库没有依赖锁定文件，先准备 Python 3.10+ 环境并安装四个第三方包：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pandas matplotlib plotly
```

Windows 使用 cmd 时运行 `.venv\Scripts\activate.bat`，使用 PowerShell 时运行 `.\.venv\Scripts\Activate.ps1`。

将每个品种的分钟行情保存为 `data/<SYMBOL>.csv`，例如 `data/CU.csv`；除随仓库保留的 `LC.csv` 外，其余数据需要自行放入。CSV 至少包含以下字段：

```text
datetime,open,high,low,close,volume,total_turnover,open_interest
```

### Chapter 1：11—14 号因子

一键完成分钟数据聚合、全品种因子计算、汇总和组合信号绘图：

```bash
python3 code/chapter1/run/run_all.py
```

因子完成后，可运行信号胜率评估；加 `--skip-plots` 可只生成表格：

```bash
python3 code/chapter1/backtest/evaluate_signal_win_rate.py
python3 code/chapter1/backtest/evaluate_signal_win_rate.py --skip-plots
```

只准备 Chapter 1 日频数据或只重新生成组合图时分别运行：

```bash
python3 code/chapter1/00_prepare_data.py
python3 code/chapter1/plot/plot_combined_signals.py
```

直接运行 `11price...py` 至 `14uptrend...py` 时必须自行设置 `SYMBOL`；通常应使用 `run_all.py`，它会自动设置所需环境变量和输出目录。

### Chapter 2：21—24 号因子

先生成缓存，再选择因子运行：

```bash
python3 code/chapter2/prepare_data.py --frequency daily
python3 code/chapter2/prepare_data.py --frequency hourly

python3 code/chapter2/run_factor.py --factor 21
python3 code/chapter2/run_factor.py --factor 22
python3 code/chapter2/run_factor.py --factor 23
python3 code/chapter2/run_factor.py --factor 24
python3 code/chapter2/run_factor.py --factor 24_rolling
```

21、23 号因子只需要日频缓存；22、24 和 `24_rolling` 同时需要日频与小时频缓存。滚动回测完成后，可追加生成持仓时长统计：

```bash
python3 code/chapter2/rolling/factor_24_rolling_holding_time.py
```

### Chapter 3：龙头—跟随研究

基于分钟行情的逐小时识别、简单做空回测和可视化按以下顺序运行：

```bash
python3 code/chapter3/run/run_identification.py
python3 code/chapter3/backtest/run_short_backtest.py --fee-rate 0.0001
python3 code/chapter3/plot/visualize.py
python3 code/chapter3/plot/plot_backtest_equity.py
python3 code/chapter3/plot/plot_backtest_prices.py
```

三参数网格会直接读取 `data/`，不依赖前面的识别结果：

```bash
python3 code/chapter3/backtest/run_short_grid.py --fee-rates 0,0.0001
```

31、32 号日频因子需要先生成 Chapter 2 的日频缓存：

```bash
python3 code/chapter2/prepare_data.py --frequency daily
python3 code/chapter3/factors/factor31.py
python3 code/chapter3/factors/factor32.py
python3 code/chapter3/plot/plot_factor31_equity.py
python3 code/chapter3/backtest/tune_factor32.py
```

`tune_factor32.py` 计算量较大；`common/process.py` 是不被其他主流程读取的辅助预处理入口，如需生成 `results/chapter3/processed/` 可单独运行：

```bash
python3 code/chapter3/common/process.py
```

所有运行结果都会写入 `results/chapter1/`、`results/chapter2/` 或 `results/chapter3/`。

## 3. 每个文件的作用

### 根目录

| 文件 | 一句话说明 |
| --- | --- |
| `.gitignore` | 声明原始数据、研究结果、系统文件、虚拟环境、工具缓存和本地调参草稿的忽略规则。 |
| `README.md` | 说明项目结构、运行方式以及版本库中每个文件的用途。 |

### `code/chapter1/`

| 文件 | 一句话说明 |
| --- | --- |
| `code/chapter1/00_prepare_data.py` | 把 `data/*.csv` 分钟行情按自然日聚合为 Chapter 1 日频表并计算投机度。 |
| `code/chapter1/backtest/evaluate_signal_win_rate.py` | 提供信号簇事件胜率评估的命令行入口并校验运行参数。 |
| `code/chapter1/backtest/evaluate_signal_win_rate_core.py` | 负责发现因子文件、合并信号簇、判断未来回撤命中并生成基准及分组统计。 |
| `code/chapter1/backtest/evaluate_signal_win_rate_outputs.py` | 负责输出胜率事件明细、汇总 CSV、信号图和终端报告。 |
| `code/chapter1/factors/11price_up_volume_oi_surge.py` | 计算“价格强势、成交放大且持仓变化异常”的 11 号因子与信号。 |
| `code/chapter1/factors/12price_up_speculation_up.py` | 计算“价格和持仓均线偏强且投机度异常升高”的 12 号因子与信号。 |
| `code/chapter1/factors/13price_up_oi_down.py` | 计算“价格趋势强但高位持仓开始回落”的 13 号因子与信号。 |
| `code/chapter1/factors/14uptrend_crowded_chase.py` | 计算“上涨趋势中增仓且成交量或振幅异常”的 14 号拥挤追涨因子与信号。 |
| `code/chapter1/factors/volume_price_factor_core.py` | 提供日频读取、脚本元数据解析、历史分位、MAD 分数、均线和公共量价特征计算。 |
| `code/chapter1/factors/volume_price_factor_outputs.py` | 将因子日表、信号事件表和汇总表标准化后写入结果目录。 |
| `code/chapter1/factors/volume_price_factor_utils.py` | 为 11—14 号因子的旧导入路径重新导出核心计算和输出接口。 |
| `code/chapter1/plot/plot_combined_signals.py` | 提供 Chapter 1 多因子组合信号绘图的命令行入口。 |
| `code/chapter1/plot/plot_combined_signals_charts.py` | 生成各品种静态 PNG、交互 HTML 和组合信号总览页面。 |
| `code/chapter1/plot/plot_combined_signals_config.py` | 集中配置组合图的输入输出路径、默认因子和绘图库缓存目录。 |
| `code/chapter1/plot/plot_combined_signals_data.py` | 发现并读取各因子日表，再构造合并信号和统计数据。 |
| `code/chapter1/plot/plot_combined_signals_templates.py` | 保存组合信号 dashboard 的 HTML 模板和 Plotly 点击交互脚本。 |
| `code/chapter1/run/run_all.py` | 串联数据准备、11—14 号因子全品种运行、汇总及组合信号绘图。 |

### `code/chapter2/`

| 文件 | 一句话说明 |
| --- | --- |
| `code/chapter2/core/__init__.py` | 将 `core` 目录标记为 Chapter 2 核心工具包。 |
| `code/chapter2/core/indicators.py` | 计算均线乖离、回归斜率、历史波动率和投机度等公共指标。 |
| `code/chapter2/core/io.py` | 发现品种、读取并校验日频和小时频表以及统一写出 CSV。 |
| `code/chapter2/core/metrics.py` | 计算单利累计曲线、年化收益、最大回撤和夏普比率。 |
| `code/chapter2/core/paths.py` | 集中定义原始数据、缓存、因子结果路径并设置运行时缓存目录。 |
| `code/chapter2/core/plots.py` | 绘制信号价格图、单品种策略曲线、全品种组合图和收益汇总图。 |
| `code/chapter2/core/reports.py` | 定义标准输出路径、保存回测表格并提供空交易表结构。 |
| `code/chapter2/engines/__init__.py` | 将 `engines` 目录标记为 Chapter 2 回测引擎包。 |
| `code/chapter2/engines/backtest.py` | 从持仓状态生成含成本收益、交易明细、品种绩效和等权组合曲线。 |
| `code/chapter2/engines/daily_engine.py` | 用日频状态机在信号确认后的下一交易日开盘执行开空和平空。 |
| `code/chapter2/engines/hourly_exit_engine.py` | 将日频入场信号映射到小时行情并逐小时执行空头开平仓状态机。 |
| `code/chapter2/factors/__init__.py` | 将 `factors` 目录标记为 Chapter 2 因子配置包。 |
| `code/chapter2/factors/factor_21.py` | 配置日频“高乖离、持仓与价格回落”的 21 号因子。 |
| `code/chapter2/factors/factor_22.py` | 配置采用小时级执行和退出的 22 号因子。 |
| `code/chapter2/factors/factor_23.py` | 配置增加投机度回落过滤的日频 23 号因子。 |
| `code/chapter2/factors/factor_24.py` | 配置增加投机度过滤并采用小时级执行和退出的 24 号因子。 |
| `code/chapter2/prepare_data.py` | 按真实交易日归属夜盘并把分钟行情聚合成日频或小时频缓存。 |
| `code/chapter2/rolling/__init__.py` | 将 `rolling` 目录标记为 Chapter 2 滚动调参工具包。 |
| `code/chapter2/rolling/factor_24_param_space.py` | 围绕 24 号基准配置构造入场和退出候选参数集合。 |
| `code/chapter2/rolling/factor_24_rolling_holding_time.py` | 读取滚动回测交易表并输出已平仓交易的持有天数统计。 |
| `code/chapter2/rolling/walk_forward.py` | 按训练窗口选择参数、在测试窗口样本外回测并重建最终曲线和指标。 |
| `code/chapter2/rules/__init__.py` | 将 `rules` 目录标记为 Chapter 2 交易规则包。 |
| `code/chapter2/rules/entry_rules.py` | 定义入场参数并生成高乖离、持仓、价格及投机度回落的开空信号。 |
| `code/chapter2/rules/exit_rules.py` | 定义退出参数并判断空头成本线止损和低点反弹平仓条件。 |
| `code/chapter2/run_factor.py` | 统一加载 21—24 号因子，运行常规或滚动回测并输出表格和图形。 |

### `code/chapter3/`

| 文件 | 一句话说明 |
| --- | --- |
| `code/chapter3/__init__.py` | 将 `chapter3` 目录标记为 Python 包。 |
| `code/chapter3/backtest/__init__.py` | 将 `backtest` 目录标记为 Chapter 3 回测包。 |
| `code/chapter3/backtest/run_short_backtest.py` | 对向下跟随信号在下一根分钟 K 线开盘做空、当日收盘平仓并统计绩效。 |
| `code/chapter3/backtest/run_short_grid.py` | 对收益、持仓和相关性三个阈值进行网格回测并汇总参数稳定性。 |
| `code/chapter3/backtest/tune_factor32.py` | 分阶段粗调 factor32 的入场、关系度和退出参数并输出多种排名。 |
| `code/chapter3/common/__init__.py` | 将 `common` 目录标记为 Chapter 3 公共模块包。 |
| `code/chapter3/common/config.py` | 集中定义识别阈值、路径、字段要求和品种板块映射。 |
| `code/chapter3/common/market_data.py` | 清洗分钟行情、映射交易日并生成完整日 K 和逐小时可见日 K 快照。 |
| `code/chapter3/common/process.py` | 将分钟行情批量汇总为日 K、收益表、品种板块表和分板块 CSV。 |
| `code/chapter3/common/rules.py` | 按突破、涨跌和持仓异常及历史相关性识别龙头与跟随品种。 |
| `code/chapter3/factors/__init__.py` | 将 `factors` 目录标记为 Chapter 3 因子包。 |
| `code/chapter3/factors/factor31.py` | 使用 Chapter 2 的 23 号信号选板块龙头并做空高相关跟随品种。 |
| `code/chapter3/factors/factor32.py` | 使用可配置的 23 号式信号选龙头、按关系度筛选跟随品种并回测。 |
| `code/chapter3/plot/__init__.py` | 将 `plot` 目录标记为 Chapter 3 绘图包。 |
| `code/chapter3/plot/plot_backtest_equity.py` | 绘制向下跟随做空回测的净值曲线并标出交易日期。 |
| `code/chapter3/plot/plot_backtest_prices.py` | 绘制简单做空交易从开仓到平仓的分钟相对价格路径。 |
| `code/chapter3/plot/plot_factor31_equity.py` | 绘制 factor31 在有交易日期上的净值曲线。 |
| `code/chapter3/plot/visualize.py` | 生成龙头—跟随关系网络图和可点击的事件复盘 HTML。 |
| `code/chapter3/run/__init__.py` | 将 `run` 目录标记为 Chapter 3 命令行入口包。 |
| `code/chapter3/run/run_identification.py` | 提供逐小时龙头—跟随识别入口并写出识别结果和日 K 中间表。 |

### `data/` 与 `docs/`

| 文件 | 一句话说明 |
| --- | --- |
| `data/LC.csv` | 保存碳酸锂（LC）的分钟级 OHLC、成交量、成交额和持仓量原始行情。 |
| `docs/chapter1_1_notes.md` | 说明 Chapter 1 的数据口径、公共指标和 11—14 号因子信号逻辑。 |
| `docs/chapter1_2_notes.md` | 说明 Chapter 1 信号簇胜率事件研究的动态阈值、分组和胜负口径。 |
| `docs/chapter2_1_notes.md` | 说明 Chapter 2 高乖离持仓回落策略的 21—23 号开空和平空逻辑。 |
| `docs/chapter2_2_notes.md` | 说明 Chapter 2 的 21—24 号因子公式、参数差异和滚动筛选逻辑。 |
| `docs/chapter3_1_notes.md` | 说明逐小时识别期货龙头和同板块跟随品种的规则、阈值及流程。 |
| `docs/chapter3_2_notes.md` | 说明 factor31、factor32 的龙头—跟随逻辑、交易执行和研究结果。 |

### `useful_plots/`

| 文件 | 一句话说明 |
| --- | --- |
| `useful_plots/SA_21_high_bias_oi_drop_signal_on_price.png` | 展示纯碱（SA）21 号因子信号在价格走势上的位置。 |
| `useful_plots/SF_21_high_bias_oi_drop_signal_on_price.png` | 展示硅铁（SF）21 号因子信号在价格走势上的位置。 |
| `useful_plots/SI_21_high_bias_oi_drop_signal_on_price.png` | 展示工业硅（SI）21 号因子信号在价格走势上的位置。 |
| `useful_plots/V_21_high_bias_oi_drop_signal_on_price.png` | 展示 PVC（V）21 号因子信号在价格走势上的位置。 |
| `useful_plots/chapter2_factor_21_all_symbols_summary.png` | 保存 21 号因子全品种等权组合净值概览图。 |
| `useful_plots/chapter2_factor_21_return_summary.png` | 保存 21 号因子各品种收益表现汇总图。 |
| `useful_plots/chapter2_factor_22_all_symbols_summary.png` | 保存 22 号因子全品种等权组合净值概览图。 |
| `useful_plots/chapter2_factor_22_return_summary.png` | 保存 22 号因子各品种收益表现汇总图。 |
| `useful_plots/chapter2_factor_23_all_symbols_summary.png` | 保存 23 号因子全品种等权组合净值概览图。 |
| `useful_plots/chapter2_factor_23_return_summary.png` | 保存 23 号因子各品种收益表现汇总图。 |
| `useful_plots/chapter2_factor_24_all_symbols_summary.png` | 保存 24 号因子全品种等权组合净值概览图。 |
| `useful_plots/chapter2_factor_24_return_summary.png` | 保存 24 号因子各品种收益表现汇总图。 |
| `useful_plots/chapter2_factor_24_rolling_all_symbols_summary.png` | 保存滚动选参 24 号因子的全品种等权组合净值概览图。 |
| `useful_plots/chapter2_factor_24_rolling_return_summary.png` | 保存滚动选参 24 号因子的各品种收益表现汇总图。 |
| `useful_plots/combined_factor_signals_dashboard.html` | 保存 Chapter 1 多因子组合信号的离线交互总览页面。 |
| `useful_plots/event_review.html` | 保存 Chapter 3 龙头—跟随事件的离线交互复盘页面。 |
| `useful_plots/plotly.min.js` | 为离线交互图提供 Plotly 的压缩 JavaScript 运行库。 |

### 本地输入、运行产物和缓存

这些文件不受 Git 跟踪，但每种文件的作用如下：

| 文件规则 | 一句话说明 |
| --- | --- |
| `data/<SYMBOL>.csv` | 每个文件保存文件名所对应期货品种的原始分钟行情输入。 |
| `results/chapter1/tables/<SYMBOL>_daily.csv` | 每个文件保存一个品种供 Chapter 1 使用的日频聚合行情。 |
| `results/chapter1/<因子目录>/factors/*.csv` | 每个文件保存一个品种的 Chapter 1 因子日值和特征。 |
| `results/chapter1/<因子目录>/signals/*.csv` | 每个文件保存一个品种实际触发的 Chapter 1 信号事件。 |
| `results/chapter1/<因子目录>/summary/*.csv` | 每个文件保存单品种或全品种的 Chapter 1 因子摘要。 |
| `results/chapter1/combined/**` | 各文件分别保存多因子信号统计、单品种静态/交互图和总览页面。 |
| `results/chapter1/evaluation/**` | 各文件分别保存信号簇事件、胜率汇总和事件价格图。 |
| `results/chapter2/tables/daily/<SYMBOL>_daily.csv` | 每个文件保存一个品种按真实交易日聚合的日频行情缓存。 |
| `results/chapter2/tables/hourly/<SYMBOL>_hourly.csv` | 每个文件保存一个品种按真实交易日标记的小时行情缓存。 |
| `results/chapter2/factor_*/tables/*.csv` | 各文件分别保存品种指标、交易明细、收益曲线、组合曲线或滚动选参结果。 |
| `results/chapter2/factor_*/figures/signals/*.png` | 每张图展示一个品种的因子开平仓信号位置。 |
| `results/chapter2/factor_*/figures/strategy/*.png` | 每张图展示一个品种的策略与基准累计收益。 |
| `results/chapter2/factor_*/figures/*summary.png` | 每张图展示对应因子的全品种组合或收益横向汇总。 |
| `results/chapter3/identification/*.csv` | 各文件分别保存龙头结果、跟随结果和识别使用的日 K。 |
| `results/chapter3/short_backtest/*.csv` | 各文件分别保存简单做空策略的交易、日收益和绩效。 |
| `results/chapter3/short_grid/*.csv` | 各文件分别保存参数网格指标、排名、参数摘要和交易明细。 |
| `results/chapter3/factor31/*.csv` | 各文件分别保存 factor31 的龙头、跟随、交易、日收益和绩效。 |
| `results/chapter3/factor32/*.csv` | 各文件分别保存 factor32 的龙头、跟随、交易、日收益、绩效和参数。 |
| `results/chapter3/factor32_tuning/*.csv` | 各文件分别保存 factor32 全部调参结果及不同评价口径的排名。 |
| `results/chapter3/figures/*` | 各文件分别保存净值图、价格路径图、龙头—跟随网络图或事件复盘网页。 |
| `results/chapter3/processed/**` | 各文件保存辅助预处理生成的品种表、日 K、收益表和分板块数据。 |
| `**/__pycache__/*`、`*.pyc`、`*.nbc`、`*.nbi` | 这些文件是 Python 或 Numba 自动生成的字节码及编译缓存。 |
| `.matplotlib/**`、`results/**/.matplotlib-cache/**` | 这些文件是 Matplotlib 自动生成的字体和绘图缓存。 |
| `.pytest_cache/**` | 这些文件是 Pytest 自动生成的测试发现和上次运行状态缓存。 |
| `.DS_Store` | 这些文件是 macOS Finder 自动生成的目录显示元数据。 |
