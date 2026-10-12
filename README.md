# Wyckoff_go

基于规则的 Wyckoff 量价结构与点数图（Point & Figure）分析工具，以及围绕它搭建的离线研究、调参、标注和 AI 参考判断框架。Python 实现。

本仓库只包含策略核心与框架代码：不附带行情数据、运行结果和调参产物，也不连接账户或下单。它是一个研究工具，不构成投资建议。阶段输出是规则候选并附带依据和确认时间，不是概率；本仓库不主张任何阶段识别准确率或盈利能力。

## 灵感来源

方法本身来自 Richard D. Wyckoff：供需、因果、努力与结果三条定律，以及吸筹与派发的 A–E 阶段。

项目的起点是 Tola Moses Hector 在 MQL5 上发表的 *Automating Classic Market Methods in MQL5* 系列的两篇文章：[第 1 篇 Wyckoff Accumulation and Distribution](https://www.mql5.com/en/articles/22628) 和 [第 2 篇 Wyckoff Cause and Effect — Point and Figure Price Targets](https://www.mql5.com/en/articles/22472)。它们展示了如何把 Wyckoff 结构和点数图目标写成可执行的规则。`backtest.py` 里保留了按第 1 篇交易规则写的回测。

术语和阶段定义随后改以方法传承机构的公开教学材料为准：Roman Bogomazov 与 Michael Lipsett（Wyckoff Analytics）的 [The Wyckoff Method 教程](https://chartschool.stockcharts.com/table-of-contents/market-analysis/wyckoff-analysis-articles/the-wyckoff-method-a-tutorial)、Hank Pruden 的 CMT 讲义、Jim Forte 的 *Anatomy of a Trading Range*。点数图的刻度、反转和计数约定参照 StockCharts ChartSchool 的 Point & Figure 文档。`research/labels/` 里的九个参照点转写自 Bruce Fraser 在 StockCharts 发表的两幅 AAPL 图。

验证和调参方法参考了 Lo、Mamaysky 与 Wang（2000），Sullivan、Timmermann 与 White（1999），Bailey 等（回测过拟合概率），Cawley 与 Talbot（2010），Bergstra 与 Bengio（2012）等研究。完整书目和链接在 `doc/literature/sources.json`。原文不随仓库分发，`scripts/fetch_literature.py` 可按清单从原始地址下载公开的 PDF。

## 在起点之上做的升级

起点是一套按最近事件直接映射阶段的 EA 规则。本项目把它改成分层的、每一步都能回溯到证据和时点的分析系统。

**分析核心（`wyckoff_core/`）**

- 每个事件同时记录发生时间和确认可知时间，历史阶段必须按数据前缀逐个计算，不把最终快照回填到过去。
- 交易区间用真实 OHLCV 边界建立，有完整的生命周期：形成、确认、破位、重返、退休都显式记录，边界和身份不随滚动窗口漂移。
- 吸筹和派发两个方向的 A–E 假设并行评估，证据不足时保留未知。Spring 和 UTAD 可以缺省；单个事件不机械映射到某个阶段。
- 支点失守会追加因果失效事件，并清除依赖它的序列和计数；支持有自己的生命周期。
- 三条定律各自给出可检查的证据。九项测试和五步法做成检查清单，缺少基准、历史目标或交易计划时返回不可用，不制造通过条件。
- 相对强弱和市场趋势只在基准与标的同时间覆盖时给出，不前向填充，也不拿过期的基准窗口代替当前判断。
- 水平跨度计数只在等宽网格且有有效锚点时输出，不用中点凑目标。

**点数图（`pnf.py`）**

- 固定箱格（按 ATR 中位数取值）、百分比、传统三种刻度共用一个增量构造引擎。
- 对 OHLC 无法还原日内先后顺序的问题给出明确约定。
- 描述性结构、垂直计数目标，可选绘图。

**数据（`market_data.py`）**

- 统一的 OHLCV 契约：时间严格递增且唯一，价格为有限正数并符合 OHLC 关系，缺口直接报错并列出日期。
- 本地 CSV 和 Yahoo 下载两种来源，分析库本身不下载、不绘图、不写文件。

**研究框架（`research/`）**

- 分市场研究：美股和 Binance 现货分开处理，逐根前缀回放，并生成不依赖外部库的离线交互回放页面。
- 长历史：下载时保存证券身份和完整性清单，逐日导出阶段和事件账本，再与独立专家标注对照，弃答计为不一致。
- 逐标调参（`research/tuning/`，命令 `wyckoff-tune`）：31 项参数的分市场搜索空间，Optuna TPE 与随机搜索对照，时序切分与信息区间清除，盲标包的生成与导入，双人独立标注的一致性检查与第三人裁决，冻结留出验证和成对聚类自助法。
- AI 参考判断（`research/llm_*.py`）：数据包去除标的、日期并归一价格后才交给模型；提示词和输出合同都带版本；每条论断必须逐字引用输入里的数值；v3 合同要求模型用 session 编号声明区间、离开、支点和支持，由程序用原始 bars 核对，把问题分成事实错误和证据缺项两类。请求有预算上限，只有可用性失败才换渠道，不做语义重试。
- 早期实验保留在 `backtest.py` 和 `research/` 的其余模块里：事件研究、止损止盈扫描、波动率仓位管理、walk-forward 调参。

## 目录

| 路径 | 内容 |
| --- | --- |
| `wyckoff_core/` | 因果特征、区间生命周期、量价事件、阶段、三定律、准备度检查、水平计数 |
| `pnf.py` | 点数图构造、描述性结构、计数目标、绘图 |
| `market_data.py` | OHLCV 契约、CSV 读取、行情下载 |
| `analysis.py`、`wyckoff.py` | 组合分析、JSON 与文字报告、命令行入口 |
| `backtest.py` | 早期的方向择时回测与参数网格 |
| `research/` | 分市场研究、长历史、阶段核查、回放页面、AI 参考判断、早期实验 |
| `research/tuning/` | 调参、标注、切分、留出验证 |
| `research/labels/` | 独立专家参照点及其来源说明 |
| `scripts/`、`doc/literature/` | 文献清单与下载脚本 |
| `tests/` | 测试 |

## 使用

需要 Python 3.11 或以上。

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[yahoo,plot,dev]'

# 从 Yahoo 取数分析
.venv/bin/python -m analysis SPY --scale atr

# 用本地 CSV，可同时给出市场基准
.venv/bin/python -m analysis SPY --csv prices.csv --format json --output output/SPY.json
.venv/bin/python -m analysis AAPL --csv aapl.csv --benchmark-csv spy.csv
.venv/bin/python -m analysis SPY --csv prices.csv --scale atr --scale pct --scale trad --chart-dir output/charts

.venv/bin/pytest -q
.venv/bin/ruff check .
```

CSV 列为 `Date` 或 `Datetime`，以及 `Open,High,Low,Close,Volume`。安装后 `wyckoff-analyze` 等价于 `python -m analysis`。

作为库调用：

```python
from analysis import analyze_frame, to_json
from market_data import read_csv

result = analyze_frame(read_csv("prices.csv", "SPY"), kinds=["atr", "pct"])
print(to_json(result))
```

研究和调参入口需要额外依赖，并且都要显式运行；导入模块和 `--help` 不会下载或写文件。

```bash
uv pip install --python .venv/bin/python -e '.[yahoo,research,tuning,dev]'
.venv/bin/python -m research.history_data --start 2006-09-30 --end 2026-09-30 --output output/long-history/run
.venv/bin/python -m research.long_study --history output/long-history/run
.venv/bin/python -m research.phase_validation --study output/long-history/run/study
.venv/bin/wyckoff-tune --help
```

AI 参考判断需要在本地 `.env` 里提供模型服务的密钥，该文件不入库。准备数据包和提示词是离线的；真实请求会产生费用，需要显式执行。

## 方法边界

- 阶段、事件和水平拥挤区都是规则候选。`confidence` 是证据强弱的定性标签，不是概率。
- 输入默认只含已完成的 K 线。日线事件最早在该根收盘后可用。
- OHLC 不含日内顺序，点数图采用延伸优先、首个双向穿箱按收盘位置定方向的约定。
- ATR 中位数箱格是历史快照，重新估算会改变历史图形。
- 趋势参考线是启发式的，没有实现完整的自动趋势线切换规则。
- OHLCV 只能间接反映供需，不能当作订单流或机构持仓。
- 参数是工程启发式。测试通过和事实核查通过只说明实现与所声明的规则一致，不说明阶段判断正确。

## 许可

本项目代码以 [MIT License](LICENSE) 发布。

引用的教程、论文和图表版权归各自作者所有，本仓库只保存书目信息和链接，不分发原文。`research/labels/independent_phase_labels.json` 记录的是从公开图表转写的日期与阶段，并注明作者和出处，不含原图。
