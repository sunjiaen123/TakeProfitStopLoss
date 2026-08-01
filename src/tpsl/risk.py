from __future__ import annotations

import math
from dataclasses import dataclass

from .config import RecommendationConfig


@dataclass(frozen=True)
class RiskDecision:
    take_profit_price: float | None
    take_profit_enabled: bool
    take_profit_reason: str
    stop_trigger_price: float
    stop_limit_price: float
    risk_reward_ratio: float
    confidence: float
    reason: str


def round_to_tick(value: float, tick: float, direction: str = "nearest") -> float:
    units = value / tick
    if direction == "down":
        rounded = math.floor(units + 1e-10)
    elif direction == "up":
        rounded = math.ceil(units - 1e-10)
    else:
        rounded = round(units)
    return round(rounded * tick, 6)


def make_risk_decision(
    *,
    close: float,
    avg_cost: float,
    max_loss_pct: float,
    current_stop: float | None,
    predicted_high_return: float,
    predicted_low_return: float,
    similar_high_return: float,
    similar_low_return: float,
    atr_14: float,
    similarity_distance: float,
    sample_count: int,
    config: RecommendationConfig,
) -> RiskDecision:
    blended_high = 0.70 * predicted_high_return + 0.30 * similar_high_return
    blended_low = 0.70 * predicted_low_return + 0.30 * similar_low_return
    take_return = min(
        max(blended_high, config.minimum_take_profit_pct),
        config.maximum_take_profit_pct,
    )
    take_profit = close * (1 + take_return)
    cost_profit_floor = avg_cost * (1 + config.minimum_profit_over_cost_pct)
    cost_floor_applied = take_profit < cost_profit_floor
    take_profit = max(take_profit, cost_profit_floor)

    model_stop = close * (1 + min(blended_low, -config.minimum_stop_gap_pct))
    atr_stop = close - config.atr_stop_multiplier * atr_14
    cost_floor = avg_cost * (1 - max_loss_pct)
    # The minimum gap is a noise filter for model/ATR stops. It must not
    # push the hard cost-loss floor or an existing ratcheted stop downward.
    soft_stop = min(
        max(model_stop, atr_stop),
        close * (1 - config.minimum_stop_gap_pct),
    )
    stop_candidates = [soft_stop, cost_floor]
    if current_stop is not None and math.isfinite(current_stop):
        stop_candidates.append(current_stop)
    stop_trigger = max(stop_candidates)
    # If a hard stop has already been breached, use the current close as an
    # immediate trigger instead of emitting an invalid trigger above market.
    stop_trigger = min(stop_trigger, close)

    if config.default_daily_price_limit_pct > 0:
        limit_pct = config.default_daily_price_limit_pct
        take_profit = min(take_profit, close * (1 + limit_pct))
        stop_trigger = max(stop_trigger, close * (1 - limit_pct))

    take_profit = round_to_tick(take_profit, config.price_tick, "nearest")
    stop_trigger = round_to_tick(stop_trigger, config.price_tick, "down")
    stop_limit = round_to_tick(
        stop_trigger * (1 - config.stop_limit_slippage_pct),
        config.price_tick,
        "down",
    )

    reward = max(0.0, take_profit - close)
    risk = max(config.price_tick, close - stop_trigger)
    risk_reward = reward / risk

    distance_score = 1 / (1 + max(0.0, similarity_distance))
    sample_score = min(1.0, sample_count / max(1, config.similarity_top_k))
    agreement_gap = abs(predicted_high_return - similar_high_return) + abs(
        predicted_low_return - similar_low_return
    )
    agreement_score = max(0.0, 1 - agreement_gap / 0.10)
    confidence = max(
        0.0,
        min(1.0, 0.40 * distance_score + 0.30 * sample_score + 0.30 * agreement_score),
    )

    warnings: list[str] = []
    take_profit_blocks: list[str] = []
    if risk_reward < config.minimum_risk_reward:
        warnings.append(
            f"盈亏比 {risk_reward:.2f} 低于阈值 {config.minimum_risk_reward:.2f}"
        )
        take_profit_blocks.append("盈亏比不足")
    if confidence < config.minimum_take_profit_confidence:
        take_profit_blocks.append(
            f"置信度 {confidence:.2f} 低于阈值 "
            f"{config.minimum_take_profit_confidence:.2f}"
        )
    if predicted_high_return <= 0 or similar_high_return <= 0:
        take_profit_blocks.append("模型与相似形态未共同指向上涨")
    if stop_trigger <= cost_floor + config.price_tick:
        warnings.append("止损受最大持仓亏损约束")
    if current_stop is not None and math.isfinite(current_stop):
        warnings.append("已应用现有止损价，不下调止损")
    if cost_floor_applied:
        warnings.append("模型目标不足以覆盖持仓成本，止盈价已抬高至成本保护线")
        take_profit_blocks.append("模型目标不足以覆盖持仓成本")
    take_profit_enabled = not take_profit_blocks
    take_profit_reason = (
        "满足止盈条件"
        if take_profit_enabled
        else "暂不设置止盈：" + "；".join(take_profit_blocks)
    )
    reason = (
        f"模型次日高/低收益 {predicted_high_return:.2%}/{predicted_low_return:.2%}；"
        f"相似形态 {similar_high_return:.2%}/{similar_low_return:.2%}；"
        f"ATR14={atr_14:.4f}"
    )
    if warnings:
        reason += "；" + "；".join(warnings)

    return RiskDecision(
        take_profit_price=take_profit if take_profit_enabled else None,
        take_profit_enabled=take_profit_enabled,
        take_profit_reason=take_profit_reason,
        stop_trigger_price=stop_trigger,
        stop_limit_price=stop_limit,
        risk_reward_ratio=risk_reward,
        confidence=confidence,
        reason=reason,
    )
