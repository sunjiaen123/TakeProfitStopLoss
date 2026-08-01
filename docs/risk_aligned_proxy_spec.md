# Risk-Aligned Proxy 动态止损模型 — 实现规范 (v1)

参谋长出具，交 Codex 实现。目标：替换当前 `risk_fit.py` 的 "per-path argmax 选 gap 再回归" 路线，
改为 **`score = f(features, gap)` 代理模型**，且训练目标、晋级门、最终评测三者闭环对齐。

适用前提：本轮只验证 "是否存在 form→gap 的可学信号"。不扩模型输出、不扩候选区间、不改硬亏损线。

---

## 0. 核心设计原则（先读，决定下面一切）

**线性等价性**：per-path 分数若对模拟输出线性，则
`mean_over_paths( score(path, g) ) == J(g)`，其中 `J` 是聚合目标。
推论：

1. 权重 λ 不靠拍脑袋——它就是聚合目标 `J` 的权重，按晋级门的 max_loss 容忍度标定（见 §2）。
2. 基线不靠另选——`g* = argmax_g J(g)`（在固定 gap 网格上）就是 gate 对齐的最优固定 gap。
3. **ML 唯一要证明的事**：按状态变化的 gap 策略，能在 held-out 上按晋级门打过单一 `g*`，且超过噪声。
   打不过 → 冻结 ML 线（见 §6 go/no-go）。

---

## 1. per-path 风险对齐分数（精确定义）

对每条 path（一个 `(symbol, entry_date, holding_days)`）与每个候选 gap，
复用 `_simulate_path`（stop_tuning.py）已产出的字段，定义：

```
raw_score(path, g) =
      strategy_return                       # 模拟最终收益
    + λ_loss      * strategy_max_loss        # 注意 max_loss ≤ 0，直接相加即"越深越扣"
    - λ_whip      * whipsaw                  # 现有二值字段（stopped 且 hold-strat ≥ 2%）
    - λ_unfilled  * (stop_limit_unfilled_count > 0)
    [ + λ_dd  * strategy_max_drawdown ]      # 可选：若也想压回撤，max_dd ≤ 0，权重小
```

设计要点（避免重蹈 `_candidate_score` 的覆辙）：

- **用全幅 `strategy_max_loss` 线性项，不要 `max(0, -maxloss - floor)` 铰链**。
  铰链在正常区间恒为 0（被 5% 硬线兜住），没有梯度——这正是旧分数失效的原因。
  全幅线性项保证每个 gap 都有"尾部成本"信号。
  （若想在超预算处额外加压，可在线性项之外再叠一个小权重铰链，但不要只用铰链。）
- **不要把 excess_return 放进 per-path 分数**：`hold_return` 在一条 path 内是常数
  （stop_tuning.py:278），它对 path 内选 gap 是 no-op，只会污染跨 path 的回归 level。
- **p05 不进 per-path 分数**：它是跨路径尾分位，per-path 编码不了；靠 `λ_loss` 重罚单路径
  深亏在聚合后间接顶上去，最终由晋级门校验。

---

## 2. λ 标定方法（从晋级门反推，不拍脑袋）

聚合目标（由 §0 线性等价直接得到）：

```
J(g) = mean_return(g) + λ_loss * mean_maxloss(g) - λ_whip * whipsaw_rate(g) - λ_unf * unfilled_rate(g)
```

`mean_return(g)`、`mean_maxloss(g)` 等每个固定 gap 一行，可直接从现有 candidate 模拟表
（`_build_training_samples` 里那张被丢弃的 `candidate_records`）按 gap 聚合得到（≈8 行小表）。

标定步骤：

1. 固定 `λ_whip`、`λ_unf` 为小值（建议 `λ_whip=0.05`、`λ_unf=0.03`，量级与收益相当即可，
   它们只做次级排序）。
2. **`λ_loss` 用晋级门的 max_loss 阈值反推**：
   取使 `g* = argmax_g J(g)` 的 `mean_maxloss(g*)` 恰好不劣于晋级门 max_loss 通过线的
   **最小** `λ_loss`。即在 `λ_loss ∈ {0.5,1,2,3,5,8}` 上扫，选满足
   `mean_maxloss(g*) ≥ gate_maxloss_threshold` 的最小者。
   - 解读：`λ_loss` 越大越偏紧（g* 越窄、尾部越好、收益越低）。取"刚好能通过 gate 尾部约束"的
     最小 λ，等于"在守住尾部底线的前提下尽量保留收益"，与产品定位（防大亏但别白白放弃收益）一致。
3. 记录标定结果到 metadata：`lambda_loss / lambda_whip / lambda_unf / g_star / J(g_star) / 各 gap 的 J`。

**冻结这套 λ**，§1 的 per-path 目标和这里的 `g*` 基线必须用同一套 λ，否则又回到
"A 轴优化、B 轴打分"。

---

## 3. path 内中心化（把"路径整体运气"剔出 target）

对每条 path，先按 §1 算出该 path 在所有候选 gap 上的 `raw_score`，再做 **path 内均值中心化**：

```
target(path, g) = raw_score(path, g) - mean_over_g( raw_score(path, ·) )
```

- 默认用均值中心化（保留各 gap 间的相对幅度，模型可学"高波动态下 gap 曲线更陡"）。
- 备选（更抗离群）：path 内排名归一化 `rank(g)/(n_gap-1) ∈ [0,1]`，或 path 内 z-score。
  先上均值中心化，若 target 噪声过大再切排名。
- 中心化只改训练 target，不改 §2 选 `g*` 的口径（`g*` 仍用聚合 `J` 在固定 gap 上选）。
- 推理时对一个状态评估所有候选 gap、取 argmax，path 内常数对 argmax 无影响，自洽。

---

## 4. 训练数据与模型

- **行格式**：一行一个 `(path, gap)`。特征 `X = RISK_MODEL_FEATURE_COLUMNS + [candidate_stop_gap_pct]`
  （gap 现在是输入特征）。目标 `y = target(path, g)`（§3）。
  额外保留 `path_id / symbol / entry_date / entry_month / holding_days` 仅用于分组与评测，不进 X。
- **模型**：XGBoost `reg:squarederror`。
  - **不要**对 gap 加单调约束（score 对 gap 有内部极值，不是单调）。
  - 复用现有 `config.risk_fit` 超参（n_estimators/max_depth/learning_rate）。
- 复用已算的 candidate 模拟：现在 `_build_training_samples` 已对每个 `(path, gap)` 跑了
  `_simulate_path`，只是取了 argmax 丢掉其余。改为**全部保留**，省一次重算。

---

## 5. held-out 分组 与 闭环评测口径

### 分组（防伪重复泄漏）
- **按 `entry_date` 整组切分**：同一 `(symbol, entry_date)` 衍生的 20/40/60 三条 path、
  以及它们 gap 展开后的全部行，**必须整体落在 train 或 val 一侧**，绝不跨界。
- 第一版可沿用"最后 K 个建仓月作 val"，但要确保切分键是 entry_date/entry_month、且 gap 展开行随之走。
- 后续升级为多窗口 walk-forward（train: 24Q1–24Q4 / val: 25Q1；滚动），报多窗口均值，不看单段。

### 评测（用门本身，OOS）
对 held-out 路径：proxy → 每个状态对候选 gap 网格预测 score → argmax 得每条 path 的 gap →
用该 gap（**恒定 gap/path**，与基线和 gate 同口径）跑 `_simulate_path` → `summarize_paths`，
与以下对照：

1. `g*`（§2 的 gate 对齐最优固定 gap）—— **主对手**
2. 固定 5% / 固定 10%（参考点）
3. 旧 per-path-argmax dynamic（证明新法是否更优）

**判据指标 = 晋级门那几项**（avg max_loss 不更差、p05 不更差、return 不明显更差、whipsaw 不更高）。
**禁止用回归 RMSE 当成败指标**——中心化 score 的 RMSE 只作诊断打印。

---

## 6. Go / No-Go（明确冻结规则）

ML 线**保留**当且仅当：

> 在 held-out（最好多窗口 walk-forward）上，proxy 的 argmax 策略**通过晋级门 vs `g*`**，
> 且主指标（产品定位为防大亏 → 取 avg max_loss / p05）的改善幅度 **超过噪声**
> （按 entry_month 分组 bootstrap，改善 > 1 个标准误）。

否则 **No-Go**：冻结 ML 线，产品定型为 "固定宽止损（用 `g*`）+ 持仓硬亏损线 + 可选止盈"，
`fit-risk` 降级为研究工具。

注：gate 易拒难批的非对称是对的——风控产品宁可不晋级，也不在薄证据上切生产。
单段 val 的 PASS 不足以晋级，必须多窗口。

---

## 7. 上线推理（recommend 接入）

替换 `RiskStopModel.predict_gap`：

```
对候选 gap 网格 G：
  for g in G: 预测 score_hat(features, g)
  return argmax_g score_hat   # 即 minimum_stop_gap_pct
```

- 仍受 `[min_stop_gap_pct, max_stop_gap_pct]` 约束（argmax 在网格内，自动满足）。
- 其余接线（硬亏损线优先、ratchet 只上调、reason 串记录）保持不变。
- 仍走 `--promote` 硬门：本规范的 §5/§6 产出直接喂 `promotion_diagnostics`，FAIL 默认拒绝。

---

## 8. 已知的下一步（本规范之外，勿混入）

- **train/serve 一致性**：本规范的评测仍是"恒定 gap/path"，与线上"每天重算 gap"不同。
  这是**必要非充分**校验。proxy 通过后，再单独做"每日重预测 + ratchet"的模拟对齐（Codex 计划第 4 步），
  届时才需要把持仓状态特征（浮盈浮亏、持仓天数、距成本、距硬止损、已有止损、ST/板块、涨跌停）
  加进 features 并让训练模拟也按状态每日重选 gap。
- 本轮**不做**：扩候选区间、加 ATR/滑点/止盈等多输出、改硬亏损线。

---

## 9. 给 Codex 的实现红线（自检清单）

- [ ] per-path 用全幅 `strategy_max_loss` 线性项，不用纯铰链。
- [ ] 不把 excess_return / p05 放进 per-path 分数。
- [ ] λ 按 §2 从 gate 的 max_loss 阈值反推，并写入 metadata；`g*` 与 per-path 目标用同一套 λ。
- [ ] target 做 path 内均值中心化。
- [ ] 训练行 = (path, gap)，gap 作为特征；保留 entry_date 仅供分组。
- [ ] 切分按 entry_date 整组，20/40/60 + gap 展开行不跨界。
- [ ] 评测用晋级门指标 OOS，不用 RMSE 判成败；主对手是 `g*`，不是固定 5%。
- [ ] go/no-go 用多窗口 + bootstrap 标准误；达不到就冻结 ML 线。
