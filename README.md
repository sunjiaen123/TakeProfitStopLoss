# TakeProfitStopLoss

这是一个面向 A 股持仓的次日条件单建议 MVP。系统在每个交易日收盘后：

1. 从 MySQL 读取有效持仓和历史日线；
2. 使用可配置线程池计算个股、行业和市场特征；
3. 使用 XGBoost 分位数模型预测次日最高/最低收益分布；
4. 检索历史相似形态，校验模型预测；
5. 结合持仓成本、最大亏损、ATR 和已有止损价生成建议；
6. 把结果写回 MySQL，并可同时导出 CSV 预览。

## 模型设计

第一版训练 6 个 XGBoost 分位数模型：

- 次日最高收益：30%、50%、70% 分位；
- 次日最低收益：10%、20%、50% 分位。

模型采用全市场联合训练，输入由个股走势、成交量、行业横截面和全市场状态组成。GPU 配置开启时优先使用 CUDA；CUDA 训练失败会自动回退到 CPU，并在模型元数据中记录实际设备。

为控制本地内存和 8 GB 显存，默认从全市场数据池中固定抽取 1,500 支股票训练，并强制包含当前持仓股票。可通过 `training.max_training_symbols` 调整；随机种子固定，重复训练使用相同样本。

训练并非每天执行。建议先每月训练一次；每日只执行 `recommend`。后续应根据滚动回测决定实际训练频率。

## 数据要求

参考建表脚本见 `sql/schema.sql`。如果已有业务表，可在 `config.toml` 中映射表名和字段名。`entry_date`、`max_loss_pct`、`current_stop`、`amount`、`industry` 是可选字段，可在映射中设置为空字符串。

日线价格必须采用一致口径。建议模型使用前复权行情，同时保留原始收盘价供实际条件单换算；当前 MVP 假定数据库中的 OHLC 已经是可以直接用于下单的同一价格口径。如果你的表同时保存复权价和原始价，下一版需要明确拆分。

行业字段允许为空，空值会归入 `UNKNOWN`，但这会降低板块特征质量。

## 安装

建议使用 Python 3.11 或 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
Copy-Item config.example.toml config.toml
```

在 `config.toml` 中填写 MySQL 密码：

```toml
[database]
host = "127.0.0.1"
port = 3306
username = "root"
password = "你的密码"
```

密码中的特殊字符会由程序自动处理。

首次建库建表：

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml init-db
```

该命令会创建 `take_profit_stop_loss` 数据库和以下表：

- `stock_master`：股票、板块和涨跌停规则；
- `stock_positions`：当前持仓；
- `stock_daily_bars`：原始可交易价格、成交量和复权因子；
- `tpsl_recommendations`：每日建议；
- `tpsl_model_runs`：训练运行记录。

`config.toml` 已被加入 `.gitignore`，不会被默认提交到 Git。

初始化后，可以先手工写入一条持仓验证：

```sql
DELETE FROM take_profit_stop_loss.stock_positions
WHERE symbol = '600519.SH';

INSERT INTO take_profit_stop_loss.stock_positions
    (symbol, quantity, available_quantity, avg_cost, entry_date, max_loss_pct)
VALUES
    ('600519.SH', 100, 100, 1500.00, '2026-06-19', 0.05);
```

## 配置

默认线程数为 10：

```toml
[performance]
workers = 10
```

本地 GPU：

```toml
[training]
use_gpu = true
```

当前机器检测到 NVIDIA GeForce RTX 5060 Ti 8GB。程序只启动一个 GPU 训练流程，`workers` 用于特征计算和 CPU 训练线程，不会并行启动 10 个 GPU 模型。

## 运行

首次同步持仓股票行情，验证数据源和代码：

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml sync-data `
  --scope positions `
  --start-date 2021-01-01 `
  --end-date 2026-06-19
```

首次正式训练前同步全部 A 股：

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml sync-data `
  --scope all `
  --start-date 2021-01-01 `
  --end-date 2026-06-19
```

`data_sync.source` 支持 `auto`、`baostock` 和 `akshare`。默认 `auto`：
优先使用 BaoStock，若遇到 BaoStock 黑名单登录错误会自动切换到 AkShare。
AkShare 需要额外依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install "akshare>=1.15,<2"
```

BaoStock 使用进程级 socket，会通过 `performance.workers` 个隔离 worker 并发下载，默认 10。重复运行会先删除相同股票、相同日期区间的数据，再重新写入，因此不会产生重复日线。
已完整同步到目标截止日期的股票会自动跳过；网络失败时可直接重复执行同一命令续传。BaoStock 并发登录采用错峰和退避重试，停牌日的空成交量会以数据库 `NULL` 保存。
socket 默认 30 秒超时，避免少数异常连接让整个同步任务永久等待。

如果 BaoStock 返回 `10001011: 黑名单用户，请与管理员联系`，这是服务端拒绝登录，
继续重试无效。安装 AkShare 后保持 `source = "auto"` 即可自动 fallback；如果不安装，
可先用 `recommend --as-of-date latest` 基于本地最新完整行情生成止损建议。

首次或定期训练：

```powershell
tpsl --config config.toml train --end-date 2026-06-19
```

交易日晚间生成建议，先用 dry-run 检查。如果你不确定最新日线是否已入库，
可以使用 `latest`，程序会自动选择“所有有效持仓都有行情”的最新交易日：

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml recommend `
  --as-of-date latest `
  --dry-run `
  --output output\recommendations-latest.csv
```

确认结果后写回 MySQL：

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml recommend --as-of-date latest
```

如果要指定某个具体交易日，`as-of-date` 必须是已经完成日线入库的交易日，
而不是自然日“今天”。若报缺行情，先同步持仓日线：

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml sync-data `
  --scope positions `
  --start-date 2026-06-19 `
  --end-date today
```

## 滚动回测

第一版回测按月滚动训练。每个测试月只使用该月开始前的数据训练，模型文件保存在 `artifacts/backtests/folds`，不会覆盖生产模型。

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml backtest `
  --start-date 2026-01-05 `
  --end-date 2026-06-17 `
  --max-symbols 100 `
  --stop-order-type limit
```

回测同时比较：

- `hybrid`：分位数模型、相似形态和风控规则；
- `fixed_3_2`：固定止盈 3%、止损 2%；
- `atr`：ATR 止盈止损基准。
- `hold_close`：不设置条件单，直接持有到次日收盘。

默认使用保守成交规则：同一天同时触及止盈和止损时，按止损先发生。止损限价发生向下跳空且全天未回到限价时，标记为未成交，并按次日收盘价估值。

输出包括：

- `trades.csv`：每支股票、每个信号日的成交模拟；
- `summary.json`：胜率、盈亏比、最大回撤、Profit Factor、置信度分组等；
- MySQL 表 `tpsl_backtest_runs` 和 `tpsl_backtest_trades`。

目前没有历史持仓快照，因此回测假定每个信号日按收盘价持有股票。这用于评价条件单价格质量，不等于真实账户收益。

### 真实持仓多日止损回测

该回测读取 `stock_positions` 的真实 `entry_date`、`avg_cost` 和数量，从建仓日起逐日更新止损。止损只能上调不能下调，触发后退出；同时与完全不止损持有到截止日比较。

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml holding-backtest `
  --end-date 2026-06-18 `
  --stop-order-type limit
```

输出包括：

- `positions.csv`：每笔持仓的最终收益、持有收益、最大回撤和止损结果；
- `daily.csv`：每日建议止损、有效移动止损、次日行情和事件；
- `summary.json`：总体比较；
- MySQL 表 `tpsl_holding_backtest_runs`、`tpsl_holding_backtest_positions` 和 `tpsl_holding_backtest_daily`。

### 止损参数扫描

按每月首个交易日模拟建仓，对20、40、60个交易日持仓周期扫描 ATR 止损倍数和止损限价滑点：

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml tune-stop `
  --start-date 2026-01-05 `
  --end-date 2026-03-31 `
  --max-symbols 100
```

默认扫描：

- ATR 倍数：1.0、1.5、2.0、2.5、3.0；
- 限价滑点：0.3%、0.6%、1.0%；
- 最小止损距离：0.5%、1%、1.5%、2%、3%、5%、7%、10%；
- 同时比较止损市价模式；
- 每个历史月份只使用月初之前训练的模型。

当前默认生产参数采用平衡方案：ATR 倍数 1.5、最小止损距离 5%、
止损限价低于触发价 0.3%。纯尾部控损排名会偏向更紧的止损，
但 5% 方案能明显降低短期频繁触发，并保持较好的历史平均收益。
扫描摘要会分别输出 `best_overall`、`best_limit_order` 和
`balanced_limit_order`。

输出 `paths.csv`、`grid_results.csv` 和 `summary.json`，并写入 MySQL 表 `tpsl_stop_tuning_runs`、`tpsl_stop_tuning_results`。综合排名优先考虑5%最差收益和最大亏损，其次考虑相对持有收益、限价未成交率和过早止损率。

### 动态止损距离拟合

固定 5% 只适合作为账户级硬风控，不应该是所有形态的唯一止损距离。`fit-risk` 会基于历史路径学习“当前形态需要给多少噪音空间”，输出一个动态 `minimum_stop_gap_pct`，再交给原有风控规则计算止损价。

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml fit-risk `
  --start-date 2026-01-01 `
  --end-date 2026-03-31 `
  --max-symbols 300
```

拟合流程：

- 对每个历史建仓路径扫描多个候选止损距离，例如 0.5%、1%、2%、5%、10%；
- 对每个 `(path, gap)` 计算 risk-aligned score：收益、全幅最大亏损、过早止损、限价未成交共同决定；
- 在训练集上用显式约束选固定 baseline `g*`：平均收益不能比持有差超过配置容忍值，然后优先选平均最大亏损更浅的 gap；
- 对每条 path 做 score 均值中心化，训练 XGBoost proxy：`centered_score = f(features, candidate_stop_gap_pct)`；
- 验证时对每个 held-out path 枚举 gap，取预测 score 最高的 gap，再闭环模拟并和 `g*` 比较；
- 同时输出多窗口 walk-forward 评测；每个窗口独立训练、独立选择 `g*`，并用 bootstrap 给出动态模型相对固定 baseline 的 delta 波动；
- 默认只生成报告，不覆盖生产风控模型；
- 如果加 `--promote`，且 `promotion_diagnostics` 通过，模型会晋级到 `artifacts/risk`，`recommend` 会自动读取；
- 输出 `training_samples.csv`、`candidate_scores.csv`、`proxy_training_rows.csv`、`validation_paths.csv`、`walk_forward_summary.json`、`walk_forward_paths.csv` 和 `summary.json`。

关键配置：

- `risk_fit.proxy_lambda_loss`、`proxy_lambda_whip`、`proxy_lambda_unfilled`：proxy 训练分数里的风险厌恶权重；
- `risk_fit.fixed_baseline_min_gap_pct`：固定 baseline 的最小候选 gap；默认 0 表示沿用 `recommendation.minimum_stop_gap_pct`；
- `risk_fit.fixed_baseline_return_tolerance_pct`：固定 baseline 允许相对持有收益落后的最大幅度；
- `risk_fit.walk_forward_train_months`、`walk_forward_step_months`、`walk_forward_bootstrap_samples`：walk-forward 窗口和 bootstrap 规模。

动态模型只决定“模型/ATR 止损至少离收盘价多远”。持仓表里的 `max_loss_pct` 或 `[positions].default_max_loss_pct` 仍然是硬亏损上限；如果硬亏损线高于动态模型给出的止损，系统会优先使用硬亏损线。

当前这条 ML 路线已经冻结为研究/审计工具，不再作为当前生产晋级路线。下面的晋级入口仅保留给未来股票池或行情结构明显变化后的显式复测；日常不要使用：

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml fit-risk `
  --start-date 2026-01-01 `
  --end-date 2026-03-31 `
  --max-symbols 300 `
  --promote
```

`--promote` 会先检查 `promotion_diagnostics`。现在晋级门要求单段主验证和 walk-forward 同时通过；如果检查失败，默认拒绝覆盖生产模型，并把本次尝试写入 `artifacts/risk/promotions.jsonl` 审计日志。当前冻结状态下，不建议使用 `--force-promote`；只有明确要做隔离测试失败模型时，才使用强制晋级：

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml fit-risk `
  --start-date 2026-01-01 `
  --end-date 2026-03-31 `
  --max-symbols 300 `
  --promote `
  --force-promote
```

当前建议的生产状态是：`risk_fit.enabled = false`，`volatility_stop.enabled = false`（未配置时默认关闭），`recommendation.minimum_stop_gap_pct = 0.020`。含义是关闭动态风控层和波动差异化层，使用固定 2% 噪音止损距离，并继续保留持仓 `max_loss_pct` 或默认最大亏损线。

### 分层波动止损

`fit-volatility-stop` 用每只股票自身波动决定基准止损宽度，公式为：

```text
base_gap = clamp(k × shrunk(ATR14%), gap_min, gap_max)
```

其中 `shrunk(ATR14%)` 会按 `w=n/(n+n0)` 向同行业当日 ATR% 中位数收缩，行业不可用时退到全市场中位数。本阶段不做 ML 微调，`f=1`；模拟和推荐时将 `atr_stop_multiplier=0`，避免把 ATR 波动算两遍。硬亏损线、已有止损棘轮、限价滑点逻辑保持不变。

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml fit-volatility-stop `
  --start-date 2024-01-01 `
  --end-date 2026-03-31 `
  --max-symbols 300
```

默认只生成报告，不覆盖生产参数。当前评测口径是 shape-only：每个 walk-forward 窗口只用训练集分布校准 k，不用回测收益/回撤选宽度。一次运行会同时输出两个预注册口径：

```text
same_average_gap:
  mean(clamp(k × shrunk(ATR14%))) 对齐到目标平均 gap

same_average_stop_ratio:
  mean(clamp(k × shrunk(ATR14%)) / shrunk(ATR14%)) 对齐到固定 gap 的平均 ATR 倍数
```

目标 gap 默认是 `recommendation.minimum_stop_gap_pct`，也就是当前生产固定 2%。验证集只比较“同一量级下的波动差异化分配 vs 全员同一固定 gap”。

这个实验不再用回测收益/回撤去选择宽度，避免把 regime 噪声当成“最优止损宽度”。输出包括：

- `volatility_candidates.csv`：每个历史 path、每个 k 的波动止损模拟；
- `fixed_candidates.csv`：固定 gap baseline 模拟；
- `walk_forward_summary.json`：多窗口 walk-forward 结果；
- `summary.json`：总摘要和晋级判断。

`summary.json` 会分别给出两个口径的 pass/fail。研究判断是：两个口径都失败则冻结波动层；任一口径通过才继续研究。`--promote` 更严格，只有两个口径都通过时才会晋级，失败默认拒绝；`--force-promote` 才允许强制覆盖。当前波动层已经冻结，不建议启用生产；如果未来复测通过，生产启用方式是设置 `[volatility_stop].enabled = true`。开启后，`recommend` 会优先使用波动层；旧 `risk_fit` ML 只有在波动层关闭时才会使用。

关键配置包括 `shape_target_gap_pct`、`shrink_n0`、`gap_min/gap_max` 和 `bootstrap_samples`。`shape_target_gap_pct=0` 表示沿用 `recommendation.minimum_stop_gap_pct`。

### 当前收口结论：固定 2% 止损定型

截至 2026-06-28，围绕“止损宽度是否应该动态化”的四轮实证均未通过预注册晋级门：

| 路线 | 检验内容 | 结论 |
|---|---|---|
| proxy ML | 用形态特征学习 `gap` 条件信号 | 未体现稳定增益，结果主要受 regime 影响 |
| level 优化 | 用回测选择全局最优止损宽度/k | 最优宽度在 regime 间跳变，不稳定 |
| shape-gap | 同平均 gap 下，波动差异化 vs 固定 2% | 300 股 / 24279 条路径，pass_rate=0.2，FAIL |
| shape-stopratio | 同平均 ATR 倍数下，波动差异化 vs 固定 2% | 300 股 / 24279 条路径，pass_rate=0.0，FAIL |

因此当前产品定型为：

```text
固定 2% 软止损 + 持仓硬亏损线 + 已有止损棘轮
```

这套工具定位是控制亏损和防止过度亏损，不是收益增强系统，也不承诺跑赢持有。`fit-risk` 和 `fit-volatility-stop` 保留为 report-only 研究/审计复测台；未来只有在股票池或行情结构明显变化时，才按相同预注册判据重新评估。

暂不继续投入的方向：

- 再用回测挑“最佳止损宽度”；
- 再用当前日线特征训练形态到 gap 的 ML；
- 再用个股波动差异化替代固定 2%。

可停车、但当前不做的后续方向：

- regime 检测，在震荡/趋势之间切换策略；
- 分钟线回测，解决日内止盈止损先后顺序；
- 建仓质量本身的特征和交易选择。

### 状态机退出试验（新轴，report-only）

`backtest-exit-machine` 用来测试一条新的退出机制，不再回头调“止损宽度”。它把每日动作拆成两个：

```text
1. final_intraday_stop：次日盘中保命条件单
2. next_open_exit：收盘确认趋势破坏后，次日开盘退出
```

v2 固定参数，不做网格寻优、不切生产。核心骨架：

```text
盈利门前：逐字复用固定 2% + 硬亏损线 + 自维护棘轮，亏损单行为等同 baseline
盈利门后：达到 +1R 且自维护止损已抬到成本上方，才启用 chandelier / swing low 宽止损
确认退出：收盘跌破 MA20 缓冲和最近 swing low，次日开盘走
利润保护：按已看到的 held peak 分段抬高利润底
```

对照基准是冻结的生产镜像：

```text
固定 2% + 硬亏损线 + 自维护棘轮
```

两边都不读取数据库旧 `current_stop`，避免被旧 2% 止损遗留值污染。

运行：

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml backtest-exit-machine `
  --start-date 2024-01-01 `
  --end-date 2026-03-31 `
  --max-symbols 300
```

主评测不再把 max_loss / p05 当唯一目标。它们只作为硬约束：不能比固定 2% 明显恶化。真正要看的赢面是：

- 卖飞率是否下降，主分母用 all paths；
- 趋势保留率是否提升，赢家分类用 full path peak；
- 盈利回吐是否下降，回吐只用 held peak；
- 买错退出速度是否提升。

输出包括：

- `exit_paths.csv`：状态机和固定 2% 的逐路径配对结果；
- `walk_forward_summary.json`：多窗口 + bootstrap 结果；
- `summary.json`：总摘要、delta、判据结果。

该命令永远是 report-only，不含 `--promote`，不会修改生产配置。

### A/B：纯 2% vs 现生产止损（report-only）

`backtest-ab-stop` 用来直接比较两条当前最关键的止损候选：

```text
A = 纯 2%：同一 make_risk_decision，但 atr_stop_multiplier=0，因此 soft_stop=close×0.98
B = 现生产：同一 make_risk_decision，使用配置里的 atr_stop_multiplier=1.5
```

两边使用同批路径、同模型预测、同成交模型，只差 `atr_stop_multiplier`。这是为了回答：生产里模型/ATR 把止损放宽到 2%–5% 是否真的优于纯 2%。

运行：

```powershell
.\.venv\Scripts\tpsl.exe --config config.toml backtest-ab-stop `
  --start-date 2024-01-01 `
  --end-date 2026-03-31 `
  --max-symbols 300
```

输出在 `output/ab-stop/<run_id>/`：

- `ab_paths.csv`：A/B 逐路径配对结果；
- `walk_forward_summary.json`：多窗口 + bootstrap；
- `summary.json`：A/B 汇总、A-B delta、预注册判据。

该命令只生成报告，不写生产、不晋级、不修改配置。

## 输出解释

- `take_profit_price`：建议止盈委托价；
- `stop_trigger_price`：止损条件触发价；
- `stop_limit_price`：触发后使用的限价，低于触发价以留出成交空间；
- `dynamic_stop_gap_pct`：动态风控模型预测的最小止损距离；为空表示未使用动态模型；
- `risk_reward_ratio`：基于当日收盘价的预期盈亏比；
- `confidence`：模型、相似样本数量和相似距离组成的内部评分，不是收益概率；
- `risk_model_version`：动态风控模型版本；
- `reason`：预测分布、相似形态和风控约束摘要。

止盈默认使用次日最高收益的 70% 分位，并且不得低于持仓成本上方 0.3%；该成本保护比例可通过 `recommendation.minimum_profit_over_cost_pct` 调整。

止损属于强制风险控制，每个持仓交易日都会输出。止盈是可选项，只有同时满足以下条件才输出价格：

- 预期盈亏比不低于 `recommendation.minimum_risk_reward`；
- 置信度不低于 `recommendation.minimum_take_profit_confidence`；
- 模型与相似历史形态都指向上涨；
- 模型目标足以覆盖持仓成本保护线。

条件不足时，`take_profit_enabled=0`、`take_profit_price=NULL`，并在 `take_profit_reason` 中写明原因；止损建议不受影响。

## 当前边界

- 日线无法判断同一天内止盈和止损哪个先触发，严谨回测需要分钟线；
- 暂未处理 ST、科创板、创业板、北交所各自的涨跌停规则；
- 暂未处理除权除息导致的持仓成本和条件单价格调整；
- 当前回测股票池来自现有上市股票数据，存在幸存者偏差；
- 当前回测是假定持仓回测，不包含真实历史仓位、佣金和印花税；
- 建议价格是模型研究输出，不保证成交，也不构成投资建议。
