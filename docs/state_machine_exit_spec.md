# 状态机退出策略（盘中保命单 + 收盘确认退出）— 实现规范 v1

参谋长出具，交 Codex 实现。Claude 评审。

## 0. 这是新轴，不是翻旧账

已冻结的是"单条止损的**宽度**问题"（四轮证明宽度无可利用条件信号）。本规范换一个轴：**退出的机制/流程**——盘中宽保命单 + 收盘确认退出 + 时间止损 + 分段利润保护，按持仓**状态**切换。这个轴从未测过，重开正当。

机制动因：前四轮 **whipsaw（假止损）一直是输的那项**；收盘确认正是治 whipsaw 的标准手段。

三个目标：
- 买错了：尽快走（时间止损 / 紧初始止损）
- 涨起来：别被正常回调甩出（宽 trend 止损，盘中不挂均线）
- 趋势坏了：尽量锁利润（收盘确认退出 + 分段利润底）

---

## 1. 成败判据（最重要，先锁死——别再用旧 max_loss 门）

前四轮"失败"一半是判据的锅：偏紧固定止损在 max_loss/p05 上构造性最优，任何"为拿住趋势而放宽"的设计在这两项上注定更差。所以本设计**不以 max_loss/p05 为主门**。

对照基准 = **冻结的固定 2% + 硬线**（champion）。判据：

- **硬约束（破了直接否决）**：平均 max_loss、p05 不比固定 2% 恶化超过 0.5pp（5% 硬线已保证灾难上限，这里只防系统性变坏）。
- **主赢面（要赢的）**，walk-forward 多窗口 + bootstrap：
  - 卖飞率 ↓（核心）
  - 趋势保留率 ↑
  - 盈利回吐 ↓
  - 买错退出速度 ↑（更快砍掉坏单）
- **判定**：硬约束全过 AND 主赢面里至少多数项在 ≥半数窗口稳定改善（均值改善 > 1 标准误）。

这是多目标 trade，不是单标量。报告要把每项 vs 固定 2% 的 delta 都列出来。

### 1.1 baseline 精确定义（公平镜像现生产策略）
对照 = 固定 2% + 硬线 + **自维护棘轮**，每天：
```
fixed_soft   = close × (1 − 0.02)
fixed_stop_t = max(prev_fixed_stop, fixed_soft, hard_stop)
fixed_stop_t = min(fixed_stop_t, close)
```
- **绝不读数据库旧 `current_stop`**；棘轮只用 baseline 自己维护的 `prev_fixed_stop`。
- baseline 与 state machine 在**同一批路径上配对模拟**，所有 delta 配对计算。
- 同一套盘中成交模型（`simulate_next_day`）。baseline 无收盘确认、无时间止损，纯盘中 2% 棘轮——这才是当前生产逻辑。

---

## 2. 状态机（每个持仓日，用截至当日收盘的信息，决定次日两个动作）

状态：`INITIAL`（建仓后前 N_init 天）→ `TREND`（之后）；`technical_exit` 触发即退出。

每个交易日 t 收盘后计算，输出**次日**两个动作：
1. **盘中条件止损价** `final_intraday_stop`（防极端，挂单）
2. **收盘确认退出标志** `next_day_exit`（趋势坏了，次日开盘走）

```
ATR             = atr_14_pct × close                      # 复用现有 atr_14_pct
held_peak_high  = 建仓到当前(已收盘) 最高 high            # 实盘可得，用于 profit_floor / 回吐
hard_stop       = cost × (1 − max_loss_pct)               # 默认 0.95cost，灾难底，永远在
# R（初始风险，建仓日固定）带下限护栏，防 swing low 贴成本导致 +1R 失真：
initial_stop_at_entry = min(initial_stop_at_entry, cost − tick)   # 必在成本下方
R = max(cost − initial_stop_at_entry, 0.5 × ATR_entry, tick)
```
**no-lookahead 铁律**：任何为 t+1 计算的止损/信号，只能用截至 **t 收盘**的已收盘 bar。
`held_peak_high` 只随已收盘 bar 更新，**不得用 t+1 盘中新高**回头更新当天挂单。

### 2.1 INITIAL（t ≤ N_init）：判断"买错没有"
```
atr_initial_stop      = cost − 1.5 × ATR
entry_structure_stop  = entry_swing_low − 0.3 × ATR       # 建仓附近最近 swing low
initial_stop          = max(hard_stop, atr_initial_stop, entry_structure_stop)   # 取最紧=最护
initial_stop          = min(initial_stop, cost − tick)   # 必在成本下方，防 swing low 抬到成本上
phase_stop            = initial_stop
```
**时间止损**（在 t == N_init 当日收盘评估）：
```
if 持仓以来从未达到 +1R(即 high 从未 ≥ entry + R)  AND  close < min(cost, MA10):
    next_day_exit = true   # failure_exit：买错了，次日开盘走
```

### 2.2 TREND（t > N_init）：宽止损 + 收盘确认
```
chandelier_stop   = peak_high − 3 × ATR
swing_trend_stop  = 最近确认 swing low − 0.3 × ATR
phase_stop        = max(swing_trend_stop, chandelier_stop)   # 两个"宽"里取较紧者；MA 不进盘中单
```
**收盘确认破坏**（当日收盘评估）：
```
if close < MA20 − 0.2 × ATR  AND  close < 最近 swing low:
    next_day_exit = true   # technical_exit：次日开盘走
```
（MA20 走平/下弯作为 v2 增强信号，v1 先不做。）

### 2.3 利润保护（分段，任何状态都叠加）
```
held_peak_gain = held_peak_high / cost − 1            # 注意：实盘可得的 held peak，不是 full_path
profit_floor =
    held_peak_gain < 0.08 :  无（= −inf）
    0.08 ≤ <0.15         :  cost × 1.005               # 盈利单不许变亏损单
    0.15 ≤ <0.30         :  cost + 0.30 × (held_peak_high − cost)
    ≥ 0.30               :  cost + 0.45 × (held_peak_high − cost)
```

### 2.4 合成盘中止损价（棘轮，只上调）
```
technical_stop = max(hard_stop, phase_stop, profit_floor)
final_intraday_stop = max(technical_stop, prev_final_intraday_stop)   # 自己维护棘轮
final_intraday_stop = min(final_intraday_stop, close)                 # 不报市价之上
```
**关键隔离**：`final_intraday_stop` 的棘轮只用本策略自己维护的 `prev_final_intraday_stop`，**绝不读旧 `current_stop`（2% 策略遗留，可能太紧）**。生产接入时新模块独立维护状态。

---

## 3. 日线模拟口径（必须如实，别偷价）

新写一个状态机模拟器（别硬塞进 `_simulate_path`），逐日推进，复用 `simulate_next_day`（backtest.py）的盘中成交模型。每个持仓日 t → 次日 t+1：

1. **若 t 收盘已置 `next_open_exit`**：t+1 **开盘价**退出（市价）。记 `exit_reason = failure_exit | technical_exit`。（收盘确认 = 多扛一晚，吃次日跳空，这是真实代价，必须这样模拟；命名为 `next_open_exit` 避免误解成次日收盘退。）
2. **否则**：t+1 挂 `final_intraday_stop` 盘中单 → 用 `simulate_next_day`（`next_low ≤ stop` 触发，跳空用 open 成交逻辑，limit/market 按配置）。触发则 `exit_reason = intraday_stop`。
3. **否则**：持有到 t+1 收盘，重算，进入下一天。
4. 到持有期末仍未退 → `exit_reason = hold_to_horizon`，按末日 next_close 计。

**bars 多加载 horizon+20 个交易日**（`lookahead_extend_days`），保证任何 horizon 内退出都能算 +20 日卖飞，不出现参差分母。

模拟器须记录：`exit_day, exit_price, exit_reason, R, strategy_return, max_loss(最小不利偏移), max_drawdown`，以及两个峰值：
- `held_peak_gain`：建仓→exit 期间最高浮盈（实盘可得，给 profit_floor / 回吐用）。
- `full_path_peak_gain`：建仓→horizon **完整路径**最高浮盈（不管策略何时退，给卖飞/趋势保留/赢家分类用——否则早卖会把后续大涨藏掉）。
- 退出后 `+5/+10/+20` 交易日的最高 close（算卖飞）。

---

## 4. 新评测指标（精确定义，对照固定 2%）

峰值用法分工（别用同一个 peak，否则卖飞和回吐互相污染）：卖飞/趋势保留/赢家分类用 `full_path_peak_gain`；盈利回吐用 `held_peak_gain`。

- **卖飞率（两个分母都报，主门看 all_paths）**：
  - `sell_fly_rate_exited_only`（诊断）：在已退出路径里，`max(close[exit+1..exit+H]) ≥ exit_price×(1+thr)` 的比例。
  - `sell_fly_rate_all_paths`（**主门**）：分子同上，分母=全部路径。避免"少退出"策略刷低分母的选择偏差。
  - H∈{5,10,20}，thr∈{5%,10%}；某 H 在 horizon 内不可观测的路径剔出该 H 的分母。越低越好。
- **趋势保留率**：在赢家路径（`full_path_peak_gain ≥ 0.15`）里，`strategy_return ≥ 0.5 × full_path_peak_gain` 的比例。越高越好。
- **盈利回吐**：在 `held_peak_gain > 0` 的路径里，`mean((held_peak_gain − strategy_return) / held_peak_gain)`。衡量"看到的利润守住多少"，**用 held peak**。越低越好。
- **买错退出速度**：在坏单路径（`full_path 到 horizon 收益 < 0`）里，`exit_day ≤ N_init` 的比例（越高越好）+ 坏单平均 exit_day（越小越好）。
- **平均持仓天数**：所有路径 exit_day 均值。
- 同时仍报 max_loss / p05 / 收益 / whipsaw（作 §1 硬约束校验，非主门）。

---

## 5. 护栏（用四轮血换来的，不可谈判）

1. **参数一律用 §6 的常识默认值，禁止回测网格搜调参**。本设计不是"拟合"，是验证"结构"。一旦调参就是 regime 过拟合重演。
2. **最小骨架先行**：v1 只做下面这套；影线/放量/K线结构识别、MA 斜率、多级利润再细分 → 全部 v2，骨架赢了再加，每条增量归因。
3. **walk-forward 多窗口 + bootstrap**，对照冻结的固定 2%。
4. **report-only**；生产配置 `[exit_machine].enabled` 默认 false，本期不切生产。无参数可选，故无 `--promote` 优化；将来上线是人工 enable + 同判据复测。

---

## 6. v1 固定参数（写进配置，但不许回测优化）

```
N_init                 = 8          # 初始失败判断窗口（交易日）
progress_R             = 1.0        # 时间止损的 +1R 门槛
atr_window             = 14         # 复用 atr_14_pct
atr_initial_mult       = 1.5
entry_structure_buffer = 0.3        # × ATR
chandelier_mult        = 3.0
swing_trend_buffer     = 0.3        # × ATR
breakdown_ma           = 20         # MA20 收盘确认
breakdown_buffer       = 0.2        # × ATR
ma_fast                = 10         # 时间止损用 MA10
profit_tiers           = [(0.08, "be+0.5%"), (0.15, 0.30), (0.30, 0.45)]
max_loss_pct           = 0.05       # 复用 positions.default_max_loss_pct，硬线
swing_pivot_k          = 2          # 分型 swing：low 为前后各 k 根内最低
r_floor_atr_mult       = 0.5        # R 下限 = max(cost−initial_stop, 0.5×ATR_entry, tick)
lookahead_extend_days  = 20         # 多加载 horizon 后 20 个交易日，给卖飞 +20 用
fixed_baseline_gap     = 0.02       # baseline=固定2%+硬线+自维护棘轮（§1.1）
```

**swing low 定义**（给 Codex 明确，免得卡住）：分型低点 = `low[i]` 严格小于 `low[i−k..i−1]` 和 `low[i+1..i+k]`（k=2，需 i+k 已收盘才确认）。取"最近一个已确认分型低点"。实现简单的 rolling 实现即可；找不到则退化为"最近 10 日 rolling min"。

---

## 7. 范围（v1 做什么 / 不做什么）

**做**：§2 状态机骨架（INITIAL 紧止损 + 时间止损 / TREND 宽止损 + 收盘确认 / 分段利润底 / 棘轮 / 硬线）、§3 模拟器、§4 指标、§5 评测，新 CLI `backtest-exit-machine`（report-only，输出 vs 固定 2% 的全部指标 + walk-forward + bootstrap）。

**不做（v2+）**：长上/下影线识别、放量跌破平台、MA20 斜率信号、利润分更多级、参数寻优、生产切换。

---

## 8. Codex 自检清单

- [ ] 两个动作分离：盘中 `final_intraday_stop`（保命）+ `next_day_exit`（趋势坏次日开盘走）。
- [ ] 收盘确认退出 = **次日开盘价**成交（吃跳空），不得用当日收盘偷价。
- [ ] `final_intraday_stop` 棘轮只用本策略自维护状态，**不读旧 current_stop**。
- [ ] hard_stop 永远在 max 里（灾难底）；INITIAL 用紧 initial_stop，TREND 用宽 phase_stop。
- [ ] 时间止损：N_init 日内没到 +1R 且 close<min(cost,MA10) → failure_exit。
- [ ] 参数全用 §6 默认值，**无任何回测网格搜**。
- [ ] 新模拟器独立，复用 simulate_next_day 的成交模型；记录 exit_reason/peak/退出后窗口。
- [ ] 评测主门 = §1（硬约束 + 主赢面），**不以 max_loss/p05 当主门**；对照固定 2%；walk-forward+bootstrap。
- [ ] 记录 `held_peak_gain` 与 `full_path_peak_gain` 两个峰值；profit_floor/回吐用 held，卖飞/趋势保留/赢家用 full_path。
- [ ] 卖飞率报 exited_only + all_paths，主门看 all_paths；多加载 horizon+20 天避免参差分母。
- [ ] baseline = 固定2%+硬线+自维护棘轮（不读旧 current_stop），与 state machine 同批路径配对。
- [ ] R 带下限护栏 `max(cost−initial_stop, 0.5×ATR_entry, tick)`；initial_stop 封顶 `cost−tick`。
- [ ] 信号/止损只用截至 t 收盘的已收盘 bar，不得用 t+1 盘中信息（no-lookahead）。
- [ ] 退出信号命名 `next_open_exit`（次日开盘成交）。
- [ ] walk-forward/bootstrap 复用 volatility_stop.py 现成实现，不重写。
- [ ] report-only，`[exit_machine].enabled` 默认 false，生产不动。

---

## 9. v2 结构修正（基于 300 股 v1 失败，2026-06-28）

v1 失败诊断（代码实锤，非调参问题）：
1. `_initial_stop_for_row`(exit_machine.py:220) 的初始止损 = max(cost×0.95, cost−1.5ATR, swing−0.3ATR) ≈ **成本下方 5%**，比现生产的 2% 还宽 → 买错砍得更慢（76% vs 95.8%）、早期 p05 更深。
2. 阶段切换按**时间**（day≤n_init=initial，否则 trend），不是按盈利 → 第 9 天起即便水下也切宽跟踪 → 水下持仓被放任到 5% 硬线 → p05 −5.44% vs −2.67%、max_loss 恶化。

**v2 核心：紧到"证明自己"，宽只在成本之上。**
- **盈利门之前（未达 +1R / 成本+缓冲）**：完全用现生产紧止损 `max(prev, close×0.98, hard_stop)`，losers 行为==baseline。
- **盈利门之后（止损已抬过成本）**：切 `max(swing_trend_stop, chandelier_stop)` 宽跟踪 + 收盘确认退出。
- 阶段门从**时间**改为**盈利**；删掉 ~5% 的 ATR/structure 初始止损（它比对手还松，反向）。

**安全性质（关键）**：任何从未盈利的持仓（=尾部/losers 来源）行为完全等于现生产 → **v2 尾部构造性不劣于固定 2%**。唯一变化在已盈利的单子（拿更久）。于是干净检验"零尾部代价下能否加 trend-retention"。
**固有代价**：盈利门前用紧止损，会把"先跌后涨"的部分赢家在到达门前甩掉（少赚）；对防大亏工具是可接受方向。

**预注册判据（结构第 5 次，带停止线）**：
- 硬约束：p05/max_loss 不比固定 2% 差超 0.5pp（设计上应≈持平）。
- 要赢：趋势保留率↑、盈利回吐↓、平均收益不更差，walk-forward ≥半数窗口稳定。
- 停止线：v2 仍过不了 → **永久冻结状态机线**，产品定型固定 2%+硬线+棘轮。

**v2 红线**：
- [ ] 盈利门前逐字复用现生产 2% 止损（不是 ~5% 初始止损）。
- [ ] 阶段切换用盈利门（+1R / 成本上方），不用时间门。
- [ ] 仍不调参网格搜；仍 report-only；仍用 §1/§4 口径（full vs held peak、all_paths 卖飞分母、配对 baseline）。
```
