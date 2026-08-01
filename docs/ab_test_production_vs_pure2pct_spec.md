# A/B 测试：生产止损(B) vs 纯 2% 冠军(A) — 自包含交接 spec

给 Codex（冷启动，无需先验上下文）。本文件自含全部背景。

---

## 1. 背景：为什么要做这个测试

本项目是"A 股持仓的次日条件单建议系统"，核心目标是**防止过度亏损**（不是收益增强）。
过去一段时间，我们用 walk-forward 回测 + 预注册判据，系统性地试过 6 种"更聪明的止损"方向：
ML 预测止损距离、回测优化止损宽度、按个股波动差异化、状态机退出(v1/v2) 等。**结论：六种全部未能稳定打赢一个简单基线**，最终冻结所有动态/ML/状态机层，生产保持简单固定止损。

但复盘时发现一个**关键盲点**：

- 我们六轮回测里当 benchmark、并称为"效率前沿冠军"的那个简单策略，是 **A = 纯固定 2% + 5% 硬亏损线 + 只上调棘轮**。
- 而**生产 `recommend` 实际跑的从来不是 A**，是 **B = `src/tpsl/risk.py` 的 `make_risk_decision`**，它额外揉进了模型预测低点(`model_stop`)和 ATR 止损(`atr_stop`)，实际止损宽度在 **2%–5% 区间**浮动。
- **A 从没上过生产，B 从没被单独 A/B 验证过。** 所以"我们证明最好的(A)"≠"实际在跑的(B)"。

这个测试就是要**直接把 A 和 B 对照一次**，决定生产到底用哪个。

---

## 2. A 和 B 的精确定义（关键：只差一个参数）

两者都走同一个 `make_risk_decision`（risk.py），止损合成逻辑：
```
model_stop   = close × (1 + min(blended_low, −minimum_stop_gap_pct))   # blended_low=0.7*模型低点+0.3*相似低点
atr_stop     = close − atr_stop_multiplier × ATR14
soft_stop    = min( max(model_stop, atr_stop), close × (1 − minimum_stop_gap_pct) )
cost_floor   = avg_cost × (1 − max_loss_pct)                            # 5% 硬线
stop_trigger = max(soft_stop, cost_floor, 已有止损)                      # 棘轮只上调
stop_trigger = min(stop_trigger, close)
```

- **B（现生产）** = `atr_stop_multiplier = 1.5`，`minimum_stop_gap_pct = 0.02`。
  → model_stop / atr_stop 生效，止损 ≈ 2%–5%。

- **A（纯 2% 冠军）** = `atr_stop_multiplier = 0`，`minimum_stop_gap_pct = 0.02`。
  → **为什么 atr_mult=0 就等于纯 2%**：`atr_stop = close − 0 = close`；`max(model_stop, atr_stop) = close`（model_stop 必 < close）；`soft_stop = min(close, close×0.98) = close×0.98`。
  即 **model_stop 与 atr_stop 同时被中和，soft_stop = 纯 2%**。再叠 cost_floor + 棘轮 = A。

所以 A vs B = **同一函数、仅 `atr_stop_multiplier ∈ {0.0, 1.5}` 之差**，干净隔离"模型/ATR 拉宽到底有没有用"。

---

## 3. 实现（复用现成 harness，别新建模）

模拟器 `_simulate_path`（`src/tpsl/stop_tuning.py`）已经在内部调用 `make_risk_decision`，并接受 `atr_multiplier` / `minimum_stop_gap_pct` 参数。所以：

1. 用现成路径构建：`_entry_dates` / `_path_end_date` / `_build_paths` + `_predict_stops`（带 per-row 模型预测，volatility_stop.py / exit_machine.py 里都有调用范例），300 股、2024-01-01～2026-03-31、月度建仓、holding_days=[20,40,60]。
2. 对每条 path 跑两次 `_simulate_path`（其余参数 = 生产：`stop_order_type="limit"`，`limit_slippage_pct=0.003`，`max_loss_pct=0.05`）：
   - A：`atr_multiplier=0.0, minimum_stop_gap_pct=0.02`
   - B：`atr_multiplier=1.5, minimum_stop_gap_pct=0.02`
3. **同批路径配对**，复用 `summarize_paths` + walk-forward 切窗（`volatility_stop.py._walk_forward_windows`）+ bootstrap（`_bootstrap_deltas`），算 **A−B 配对 delta**。
   - 注：A、B 参数固定、无需"训练/选参"，walk-forward 在这里只用于**多窗口稳健性 + 离散度**，不需要 train 步。
4. **report-only**，新增一个独立命令（如 `backtest-ab-stop`）或在现有研究命令里加开关；**不写生产、不晋级**。

输出指标（A 和 B 各一份 + 配对 delta）：`average_strategy_max_loss`、`p05_strategy_return`、`average_strategy_return`、`whipsaw_rate`、卖飞率（退出后 +5/+10/+20 日又涨 5%/10%，分母用全部路径）、平均持仓天数、stopped_rate。逐窗口 + 聚合 + bootstrap pass_rate。

---

## 4. 预注册判据（先锁死，免得事后辩论）

产品是**防大亏**，所以**尾部优先**：

- **A 在 `max_loss` / `p05` 上不劣于 B，且平均收益不明显更差（差距 ≤ 0.5pp）** → **生产对齐到 A**（更紧、尾部更稳、更简单，且正好 = 已验证冠军）。
- **B 在整体上明确更好（尤其尾部不更差、收益/少卖飞更优）** → **保留 B**，结论是"模型预测低点放止损确实有用"。
- 典型情形"A 尾部好 / B 收益好"的互换 → 防大亏工具**默认取 A**，但看 delta 量级再定。

若结论是对齐 A：后续改动很小——把 `make_risk_decision` 的 soft_stop 简化为 `close×(1−minimum_stop_gap_pct)`（或生产配置 `atr_stop_multiplier=0`），并加一个"生产输出 == 纯 2% 冠军"的回归测试。**本测试阶段不做这步，先出对照结果。**

---

## 5. 护栏

- 不调参、不做网格搜（这是一次 A/B，不是寻优）。
- report-only，生产配置（`config.toml`）一律不动。
- A、B 必须同批路径、同成交模型、配对比较。
- walk-forward 多窗口 + bootstrap，报均值 + 离散度，别只看单段聚合。

跑完把对照报告（A、B 各指标 + 配对 delta + 逐窗口 + bootstrap）交回评审。
