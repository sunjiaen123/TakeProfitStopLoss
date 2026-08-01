# 分层止损（per-股波动主力 + 硬线 + 棘轮 + 可选 ML 微调）— 实现规范 (v1)

参谋长出具，交 Codex 实现。Claude 只评审，不写生产代码。

## 0. 一句话目标

把当前"全员固定 gap（生产 2%）"这一最弱的层，换成 **每只股票按自身波动算出的基准止损宽度**，
其余层（硬亏损线、棘轮、限价滑点）原样保留；全局 ML 降级为一个**有界微调系数**，本期**不实现**，
只留接口。最终是一个**职责分明、底线封死、可增量验证**的分层系统。

核心原则：止损宽度应随**个股波动**变化（高波动票宽、低波动票紧），这是止损设计里最稳的真信号；
全局形态→gap 的条件信号已被实验证明很弱，故降级。

---

## 1. 分层结构（最终合成）

沿用现有 `make_risk_decision`（risk.py）的 max/min 框架，只替换"基准宽度"层：

```
base_gap   = 每股波动宽度（§2，向板块/市场收缩补稳）           ← 新主力层
soft_stop  = close × (1 − base_gap × f)        f = ML 微调系数，本期恒为 1（§4）
final_stop = max(soft_stop, cost_floor, current_stop)          # cost_floor=avg_cost×(1−max_loss_pct)
final_stop = min(final_stop, close)                            # 跌破即触发，不报市价之上
```

各层职责（互补，缺一不可）：

| 层 | 职责 | 覆盖的失败模式 |
|---|---|---|
| ① per-股波动 → base_gap | 定止损宽窄 | 高波动票被噪音洗、低波动票太松 |
| ② 硬亏损线 cost_floor | 绝对底线，亏到 max_loss_pct 必走 | 防大亏 |
| ③ 棘轮 ratchet（只上调） | 持有期锁利 | 涨上去回吐 |
| ④ ML 微调系数 f（本期不做） | 形态/regime 下小幅收紧/放宽 | 弱条件信号 |

---

## 2. per-股波动基准宽度（live 估计器）

### 2.1 波动度量（复用现有特征，别另起炉灶）
- 主用 `atr_14_pct`（features.py 已有，每股日 ATR%）。可选再融合一个更长窗口的实现波动
  （如 60 日日收益标准差）做平滑，但**先用 ATR14% 起步**，简单稳。
- **不要**用"每股历史模拟扫 gap 取事后最优"——那是单路径事后过拟合，按股票重做一遍同样的老坑。禁止。

### 2.2 向板块/市场收缩（credibility shrinkage，补稳）
```
σ_used = w × σ_stock + (1 − w) × σ_pool
w      = n / (n + n0)          # n = 该股可用历史天数，n0 收缩常数（如 120）
σ_pool = 同行业 ATR% 中位数（行业字段 industry_code 可用；无行业则退全市场中位数）
```
目的：新上市/历史短的票（A 股次新，恰恰最该被覆盖）不会因样本少给出危险的过紧/过松值；
也防单只股票异常平静期算出过紧止损。

### 2.3 映射成 gap
```
base_gap = clamp( k × σ_used , gap_min , gap_max )
```
- `k`：唯一要标定的尺度旋钮（§5）。
- `gap_min / gap_max`：安全夹板（如 0.005 / 0.10），沿用现有 min/max_stop_gap_pct。

### 2.4 别把波动算两遍（关键）
现有 `make_risk_decision` 里已有 `atr_stop = close − atr_stop_multiplier × ATR`，本身就是每股波动。
**base_gap 与 atr_stop 不能并存重复计波动。** 二选一：
- 推荐：让 **base_gap 成为唯一波动机制**，把 `atr_stop` 这条移除或退化为同口径冗余守卫；
- 或：保留 atr_stop，但 base_gap 与其口径统一、避免双重收紧。
Codex 实现时明确选一种并在代码注释/PR 说明，我评审。

---

## 3. 接入点（改动很小）

线上 `recommend` 已有**逐股注入 per-row gap** 的现成通道：pipeline.py 里
`replace(config.recommendation, minimum_stop_gap_pct=...)`（原来喂 ML 预测的 gap）。
把它改成喂 **§2 算出的 base_gap** 即可，`make_risk_decision` 整套机制（硬线、棘轮、min(close)）不动。
模拟器 `_simulate_path`（stop_tuning.py）同理：把传入的 `minimum_stop_gap_pct` 改成"按该股 σ × k"的
per-path 值即可复用。

---

## 4. ML 微调系数 f（本期只留接口，不实现）

将来若有证据再加：`f = clamp(ML_predict(...), f_lo, f_hi)`，如 [0.8, 1.2]，默认 1.0。
有界 → ML 最坏也只能动 ±20%，永远翻不过硬线和波动基准。**本期 f 恒等于 1**，代码里留个常量/开关即可。

---

## 5. 标定 k（吸取上次 g* 角点解的教训）

- k 是**单个连续尺度**，扫网格（如 0.5…3.0 步长 0.25）。
- 每个 k：对训练集所有 path 用 `base_gap = k × σ_used` 跑现有模拟器，`summarize_paths` 聚合。
- **用显式约束目标直接挑 k，不要用线性惩罚 λ 临界法**（上次就是 λ 临界塌成 0.5% 角点）：
  > 防大亏优先：在"mean 策略收益不比 hold 差超过 τ（如 1%）"的约束内，
  > 选 **mean max_loss 最优**的 k；若多个相近，取更宽者（少 whipsaw）。
- k 是平滑尺度、目标对 k 也平滑，正常会落到内点，不会角点。把 k 与各 k 的目标表写入 metadata。

---

## 6. 验证（增量归因 + walk-forward，硬要求）

**按层叠加，每步过同一个门、跟固定 baseline 比：**

1. **第一步**：`(波动基准 + 硬线)` vs `(固定 2% + 硬线)` vs `(固定最优 gap)`。
   波动层必须赢，才算这层有价值（大概率赢，因为波动是真信号）。
2. **第二步（以后）**：加 ML 系数 f 后能否在第一步基础上再赢；赢才保留，否则 f=1 冻结。

口径：
- 复用 `_simulate_path` / `summarize_paths` / `_promotion_diagnostics`。
- **必须多窗口 walk-forward**（train: 24Q1–24Q4→val 25Q1；滚动），报各窗口均值 + 离散度，
  不看单段（单段=regime 轮盘，上次教训）。
- 判据 = 晋级门那几项（max_loss 不更差、p05 不更差、return 不明显更差、whipsaw 不更高），
  **不用回归 RMSE 判成败**。

---

## 7. 生产安全（不变）

- 默认 report-only；`--promote` 硬门（FAIL 拒绝，`--force-promote` 才覆盖）；版本化 + 审计日志。
- 当前生产保持 `risk_fit.enabled=false` + 固定 2%，直到波动层过门才切换。
- 切换方式：新增配置段（建议 `[volatility_stop]`：`enabled / lookback_days / k / k_min / k_max /
  shrink_n0 / gap_min / gap_max / pool=industry|market`），与 ML 路线解耦，互不影响。

---

## 8. 本期范围红线（别越界）

- **不做** ML 微调 f（只留接口）。
- **不做** 持仓状态特征 / regime 子模型（那是更后面的事；先让"先看见 regime"靠 walk-forward）。
- **不改** 硬亏损线、棘轮、min(close) 逻辑。
- **不用** 每股历史模拟 argmax 当估计器。
- **不并存** base_gap 与 atr_stop 的重复波动计算。

---

## 9. Codex 自检清单

- [ ] base_gap 用 `k × shrunk(ATR%)`，向行业/市场中位数收缩（credibility w=n/(n+n0)）。
- [ ] 明确处理"波动别算两遍"：base_gap 与 atr_stop 二选一并说明。
- [ ] 通过现有 per-row 注入点接入 recommend 与 _simulate_path，不重写 make_risk_decision 主体。
- [ ] k 用显式约束目标在网格上挑（防大亏优先 + return 约束），**非 λ 临界**；写入 metadata。
- [ ] 验证走多窗口 walk-forward，报均值+离散度，用晋级门判、非 RMSE。
- [ ] 第一步只比"波动 vs 固定"，ML f 恒为 1。
- [ ] report-only 默认；硬线/棘轮/--promote 门不动。
- [ ] 新配置段 `[volatility_stop]`，与 `[risk_fit]` 解耦。
```
