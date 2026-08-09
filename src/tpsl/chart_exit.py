from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from .backtest import _select_evaluation_symbols, simulate_next_day
from .config import AppConfig
from .db import load_bars
from .exit_machine import _recent_confirmed_swing_lows
from .features import build_feature_frame
from .holding_backtest import _maximum_drawdown, _predict_stops
from .risk import make_risk_decision, round_to_tick
from .stop_tuning import _entry_dates, _path_end_date


HOLD_STRATEGY = "H_hold"
PURE_2PCT_STRATEGY = "A_pure_2pct"
PRODUCTION_STRATEGY = "B_production"
CHART_STRATEGY = "C_chart_exit"
CHART_REENTRY_STRATEGY = "D_chart_reentry"
STRATEGIES = (
    HOLD_STRATEGY,
    PURE_2PCT_STRATEGY,
    PRODUCTION_STRATEGY,
    CHART_STRATEGY,
    CHART_REENTRY_STRATEGY,
)


@dataclass(frozen=True)
class ChartExitRecommendation:
    action: str
    reason: str
    stop_trigger_price: float
    stop_limit_price: float
    profile: str
    position_scale: float
    held_peak_close: float | None = None
    trend_active: bool | None = None
    trade_days: int | None = None
    diagnostic: str = ""
NO_EXIT_OUTCOMES = {
    "NO_TRIGGER_CLOSE",
    "SUSPENDED_NO_TRIGGER",
    "STOP_LIMIT_UNFILLED",
}


def _is_next_suspended(row: pd.Series) -> bool:
    value = row.get("next_is_suspended", False)
    return False if pd.isna(value) else bool(value)


def _confirmed_swing_highs(highs: pd.Series, *, pivot_k: int) -> pd.Series:
    values = pd.to_numeric(highs, errors="coerce").to_numpy(dtype=float)
    recent = np.full(len(values), np.nan, dtype=float)
    last_confirmed = np.nan
    for current in range(len(values)):
        candidate = current - int(pivot_k)
        if candidate >= int(pivot_k):
            left = values[candidate - pivot_k : candidate]
            right = values[candidate + 1 : candidate + pivot_k + 1]
            value = values[candidate]
            if (
                np.isfinite(value)
                and len(left) == pivot_k
                and len(right) == pivot_k
                and np.all(np.isfinite(left))
                and np.all(np.isfinite(right))
                and np.all(value > left)
                and np.all(value > right)
            ):
                last_confirmed = value
        recent[current] = last_confirmed
    return pd.Series(recent, index=highs.index)


def _price_series(frame: pd.DataFrame, raw_column: str, adjusted_column: str) -> pd.Series:
    source = raw_column if raw_column in frame.columns else adjusted_column
    return pd.to_numeric(frame[source], errors="coerce")


def add_chart_exit_indicators(features: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    if features.empty:
        return features.copy()

    settings = config.chart_exit
    parts: list[pd.DataFrame] = []
    for _, group in features.groupby("symbol", sort=False):
        group = group.sort_values("trade_date").copy()
        close = _price_series(group, "raw_close", "close")
        high = _price_series(group, "raw_high", "high")
        low = _price_series(group, "raw_low", "low")
        group["ma_fast"] = close.rolling(int(settings.ma_fast), min_periods=1).mean()
        group["ma_trend"] = close.rolling(int(settings.ma_trend), min_periods=1).mean()
        group["ma_long"] = close.rolling(int(settings.ma_long), min_periods=1).mean()
        group["ma_trend_slope"] = (
            group["ma_trend"]
            / group["ma_trend"].shift(int(settings.ma_slope_lookback))
            - 1.0
        )
        group["recent_swing_low"] = _recent_confirmed_swing_lows(
            low,
            pivot_k=int(settings.swing_pivot_k),
        ).fillna(low.rolling(10, min_periods=1).min())
        group["recent_swing_high"] = _confirmed_swing_highs(
            high,
            pivot_k=int(settings.swing_pivot_k),
        ).fillna(high.rolling(10, min_periods=1).max())
        group["prior_breakout_high"] = high.shift(1).rolling(
            int(settings.reentry_breakout_lookback),
            min_periods=int(settings.reentry_breakout_lookback),
        ).max()
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["symbol", "trade_date"]
    ).reset_index(drop=True)


def _prepare_chart_features(
    engine: Any,
    config: AppConfig,
    earliest_entry: date,
    load_end_date: date,
    symbols: list[str],
) -> pd.DataFrame:
    settings = config.chart_exit
    lookback = max(400, int(settings.ma_long) + 300)
    bars = load_bars(
        engine,
        config,
        earliest_entry - timedelta(days=lookback),
        load_end_date,
        symbols=symbols,
    )
    if bars.empty:
        raise RuntimeError("K线持仓回测范围没有行情数据")
    print(f"K线持仓回测特征行情读取完成：{len(bars):,} 行", flush=True)
    features = build_feature_frame(
        bars,
        workers=config.performance.workers,
        min_symbol_rows=25,
    )
    if features.empty:
        raise RuntimeError("K线持仓回测特征为空")

    features = add_chart_exit_indicators(features, config)
    parts: list[pd.DataFrame] = []
    for _, group in features.groupby("symbol", sort=False):
        group = group.sort_values("trade_date").copy()
        for source, target in (
            ("raw_open", "next_open"),
            ("raw_high", "next_high"),
            ("raw_low", "next_low"),
            ("raw_close", "next_close"),
            ("trade_date", "execution_date"),
            ("is_suspended", "next_is_suspended"),
        ):
            group[target] = group[source].shift(-1)
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["symbol", "trade_date"]
    ).reset_index(drop=True)


def _build_paths(
    predicted: pd.DataFrame,
    symbols: list[str],
    entry_dates: list[date],
    holding_days_values: tuple[int, ...],
    *,
    lookahead_days: int,
) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    grouped = {
        symbol: group.sort_values("trade_date").reset_index(drop=True)
        for symbol, group in predicted.groupby("symbol", sort=False)
        if symbol in symbols
    }
    required = [
        "raw_close",
        "raw_high",
        "raw_low",
        "atr_14_pct",
        "ma_fast",
        "ma_trend",
        "ma_long",
        "ma_trend_slope",
        "recent_swing_low",
        "prior_breakout_high",
        "predicted_high_return",
        "predicted_low_return",
        "median_high_return",
        "low_return_q20",
        "mean_distance",
        "sample_count",
        "next_open",
        "next_high",
        "next_low",
        "next_close",
        "execution_date",
    ]
    for entry_date in entry_dates:
        for symbol in symbols:
            frame = grouped.get(symbol)
            if frame is None:
                continue
            matches = frame.index[frame["trade_date"].dt.date == entry_date].tolist()
            if not matches:
                continue
            start = int(matches[0])
            for holding_days in holding_days_values:
                row_count = int(holding_days) + int(lookahead_days)
                full_rows = frame.iloc[start : start + row_count].copy()
                if len(full_rows) != row_count:
                    continue
                signal_rows = full_rows.iloc[: int(holding_days)]
                if signal_rows[required].isna().any().any():
                    continue
                paths.append(
                    {
                        "entry_date": entry_date,
                        "symbol": symbol,
                        "holding_days": int(holding_days),
                        "entry_price": float(signal_rows.iloc[0]["raw_close"]),
                        "rows": signal_rows.reset_index(drop=True),
                        "full_rows": full_rows.reset_index(drop=True),
                    }
                )
    return paths


def _path_records(paths: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path_id": int(path_id),
                "entry_date": path["entry_date"],
                "symbol": path["symbol"],
                "holding_days": int(path["holding_days"]),
            }
            for path_id, path in enumerate(paths)
        ]
    )


def _atr(row: pd.Series, close: float) -> float:
    pct = float(row.get("atr_14_pct", np.nan))
    if not np.isfinite(pct) or pct <= 0:
        return max(float(close) * 0.02, 0.01)
    return max(float(close) * pct, 0.01)


def _initial_stop(cost: float, row: pd.Series, config: AppConfig) -> float:
    settings = config.chart_exit
    tick = float(config.recommendation.price_tick)
    atr = _atr(row, cost)
    swing = float(row["recent_swing_low"])
    hard = cost * (1.0 - float(config.positions.default_max_loss_pct))
    structure = swing - float(settings.entry_structure_buffer) * atr
    raw = max(hard, structure)
    # Structure may tighten the stop, but v1 does not allow less than 2.5%
    # breathing room at entry. This is a structural guard, not an optimized value.
    stop = min(cost * (1.0 - float(settings.initial_min_gap_pct)), raw)
    stop = min(stop, cost - tick)
    return round_to_tick(stop, tick, "down")


def _profit_floor(cost: float, held_peak: float, config: AppConfig) -> float:
    settings = config.chart_exit
    gain = held_peak / cost - 1.0
    if gain < settings.profit_tier_1_gain:
        return -np.inf
    if gain < settings.profit_tier_2_gain:
        return cost * (1.0 + settings.profit_tier_1_floor_pct)
    if gain < settings.profit_tier_3_gain:
        return cost + settings.profit_tier_2_capture * (held_peak - cost)
    return cost + settings.profit_tier_3_capture * (held_peak - cost)


def _buy(
    cash: float,
    quote: float,
    config: AppConfig,
    *,
    apply_slippage: bool,
) -> tuple[float, float, float, float]:
    settings = config.chart_exit
    execution = float(quote) * (
        1.0 + (float(settings.market_slippage_pct) if apply_slippage else 0.0)
    )
    shares = float(cash) / (execution * (1.0 + float(settings.commission_rate)))
    notional = shares * execution
    fee = notional * float(settings.commission_rate)
    slippage_cost = shares * max(0.0, execution - float(quote))
    return shares, execution, fee, slippage_cost


def _sell(
    shares: float,
    quote: float,
    config: AppConfig,
    *,
    apply_slippage: bool,
) -> tuple[float, float, float, float, float]:
    settings = config.chart_exit
    execution = float(quote) * (
        1.0 - (float(settings.market_slippage_pct) if apply_slippage else 0.0)
    )
    gross = float(shares) * execution
    fee = gross * float(settings.commission_rate)
    tax = gross * float(settings.stamp_tax_rate)
    slippage_cost = float(shares) * max(0.0, float(quote) - execution)
    return gross - fee - tax, execution, fee, tax, slippage_cost


def _portfolio_mark(shares: float, cash: float, price: float) -> float:
    return float(cash) if shares <= 0 else float(shares) * float(price)


def _base_state(path: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    entry_quote = float(path["entry_price"])
    shares, cost, fee, slippage = _buy(
        1.0,
        entry_quote,
        config,
        apply_slippage=True,
    )
    return {
        "cash": 0.0,
        "shares": shares,
        "cost": cost,
        "capital_before_trade": 1.0,
        "entry_quote": entry_quote,
        "entry_date": path["entry_date"],
        "entry_index": 0,
        "entries": 1,
        "reentries": 0,
        "total_fees": fee,
        "total_taxes": 0.0,
        "total_slippage": slippage,
        "turnover": shares * cost,
        "time_in_market": 0,
        "unfilled_count": 0,
        "exit_count": 0,
        "equity": [1.0, shares * entry_quote],
        "adverse": [0.0, shares * entry_quote - 1.0],
        "max_equity": max(1.0, shares * entry_quote),
        "trades": [],
    }


def _record_sale(
    state: dict[str, Any],
    *,
    quote: float,
    row_index: int,
    exit_date: date,
    reason: str,
    outcome: str,
    config: AppConfig,
    apply_slippage: bool,
) -> None:
    cash, execution, fee, tax, slippage = _sell(
        state["shares"],
        quote,
        config,
        apply_slippage=apply_slippage,
    )
    trade_return = cash / float(state["capital_before_trade"]) - 1.0
    state["trades"].append(
        {
            "entry_date": state["entry_date"],
            "entry_index": int(state["entry_index"]),
            "entry_price": float(state["cost"]),
            "exit_date": exit_date,
            "exit_day_index": int(row_index),
            "exit_price": float(execution),
            "exit_reason": reason,
            "exit_outcome": outcome,
            "trade_return": float(trade_return),
            "is_reentry": int(state["entries"] > 1),
        }
    )
    state["cash"] = cash
    state["shares"] = 0.0
    state["total_fees"] += fee
    state["total_taxes"] += tax
    state["total_slippage"] += slippage
    # _selling_shares is populated just before calling this helper.
    state["turnover"] += execution * float(state.pop("_selling_shares", 0.0))
    state["exit_count"] += 1
    state["equity"].append(cash)
    state["adverse"].append(cash - 1.0)
    state["max_equity"] = max(float(state["max_equity"]), cash)


def _sell_position(
    state: dict[str, Any],
    *,
    quote: float,
    row_index: int,
    exit_date: date,
    reason: str,
    outcome: str,
    config: AppConfig,
    apply_slippage: bool,
) -> None:
    state["_selling_shares"] = float(state["shares"])
    _record_sale(
        state,
        quote=quote,
        row_index=row_index,
        exit_date=exit_date,
        reason=reason,
        outcome=outcome,
        config=config,
        apply_slippage=apply_slippage,
    )


def _enter_position(
    state: dict[str, Any],
    *,
    quote: float,
    entry_date: date,
    entry_index: int,
    config: AppConfig,
) -> None:
    capital = float(state["cash"])
    shares, cost, fee, slippage = _buy(
        capital,
        quote,
        config,
        apply_slippage=True,
    )
    state["cash"] = 0.0
    state["shares"] = shares
    state["cost"] = cost
    state["capital_before_trade"] = capital
    state["entry_quote"] = float(quote)
    state["entry_date"] = entry_date
    state["entry_index"] = int(entry_index)
    state["entries"] += 1
    state["reentries"] += 1
    state["total_fees"] += fee
    state["total_slippage"] += slippage
    state["turnover"] += shares * cost


def _post_exit_high_close(
    full_rows: pd.DataFrame,
    *,
    exit_day_index: int,
    days: int,
) -> float | None:
    start = int(exit_day_index)
    end = start + int(days)
    if len(full_rows) < end:
        return None
    values = pd.to_numeric(full_rows.iloc[start:end]["next_close"], errors="coerce").dropna()
    return float(values.max()) if len(values) == int(days) else None


def _finalize_result(
    *,
    path_id: int,
    path: dict[str, Any],
    strategy: str,
    state: dict[str, Any],
    hold_return: float,
    config: AppConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    final_value = float(state["cash"])
    strategy_return = final_value - 1.0
    full_rows = path["full_rows"]
    signal_rows = path["rows"]
    initial_execution = float(state.get("initial_execution_price", path["entry_price"]))
    full_peak_gain = float(signal_rows["next_high"].max()) / initial_execution - 1.0
    max_equity_gain = max(0.0, float(state["max_equity"]) - 1.0)
    retained_profit = min(max_equity_gain, max(0.0, strategy_return))
    giveback = (
        (max_equity_gain - retained_profit) / max_equity_gain
        if max_equity_gain > 0
        else 0.0
    )
    giveback_points = max(0.0, max_equity_gain - strategy_return)
    trades = list(state["trades"])
    reentry_trades = [trade for trade in trades if int(trade["is_reentry"]) == 1]
    failed_streak = 0
    max_failed_streak = 0
    for trade in trades:
        failed_streak = failed_streak + 1 if float(trade["trade_return"]) <= 0 else 0
        max_failed_streak = max(max_failed_streak, failed_streak)

    result: dict[str, Any] = {
        "path_id": int(path_id),
        "entry_date": path["entry_date"],
        "symbol": path["symbol"],
        "holding_days": int(path["holding_days"]),
        "strategy": strategy,
        "strategy_return": float(strategy_return),
        "hold_return": float(hold_return),
        "excess_return": float(strategy_return - hold_return),
        "pnl_after_cost": float(strategy_return),
        "strategy_max_drawdown": _maximum_drawdown(state["equity"]),
        "strategy_max_loss": float(min(state["adverse"])),
        "full_path_peak_gain": float(full_peak_gain),
        "profit_giveback": float(giveback),
        "profit_giveback_pct_points": float(giveback_points),
        "trend_retained": int(
            full_peak_gain >= float(config.chart_exit.winner_peak_gain)
            and strategy_return
            >= float(config.chart_exit.trend_capture_ratio) * full_peak_gain
        ),
        "stopped": int(any(t["exit_reason"] != "horizon_exit" for t in trades)),
        "exit_count": int(state["exit_count"]),
        "entry_count": int(state["entries"]),
        "reentry_count": int(state["reentries"]),
        "reentry_success_count": int(
            sum(float(trade["trade_return"]) > 0 for trade in reentry_trades)
        ),
        "time_in_market_days": int(state["time_in_market"]),
        "turnover": float(state["turnover"]),
        "transaction_cost_rate": float(
            state["total_fees"] + state["total_taxes"] + state["total_slippage"]
        ),
        "stop_limit_unfilled_count": int(state["unfilled_count"]),
        "max_consecutive_failed_entries": int(max_failed_streak),
        "win": int(strategy_return > 0),
        "whipsaw": int(strategy_return + 0.02 <= hold_return),
    }
    for days in config.chart_exit.sell_fly_days:
        for threshold in config.chart_exit.sell_fly_thresholds:
            label = f"sell_fly_{int(days)}d_{int(round(threshold * 100))}pct"
            observable = False
            flew = False
            for trade in trades:
                if trade["exit_reason"] == "horizon_exit":
                    continue
                high_close = _post_exit_high_close(
                    full_rows,
                    exit_day_index=int(trade["exit_day_index"]),
                    days=int(days),
                )
                if high_close is None:
                    continue
                observable = True
                if high_close >= float(trade["exit_price"]) * (1.0 + float(threshold)):
                    flew = True
            result[f"{label}_observable"] = int(observable)
            result[label] = int(flew)

    ledger: list[dict[str, Any]] = []
    for trade_no, trade in enumerate(trades, start=1):
        ledger.append(
            {
                "path_id": int(path_id),
                "strategy": strategy,
                "symbol": path["symbol"],
                "holding_days": int(path["holding_days"]),
                "trade_no": int(trade_no),
                **trade,
            }
        )
    return result, ledger


def _simulate_hold(
    path_id: int,
    path: dict[str, Any],
    config: AppConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = path["rows"]
    state = _base_state(path, config)
    state["initial_execution_price"] = float(state["cost"])
    for _, row in rows.iterrows():
        state["time_in_market"] += 1
        mark = _portfolio_mark(state["shares"], 0.0, float(row["next_close"]))
        adverse = _portfolio_mark(state["shares"], 0.0, float(row["next_low"]))
        state["equity"].append(mark)
        state["adverse"].append(adverse - 1.0)
        state["max_equity"] = max(
            float(state["max_equity"]),
            mark,
            _portfolio_mark(state["shares"], 0.0, float(row["next_high"])),
        )
    last = rows.iloc[-1]
    _sell_position(
        state,
        quote=float(last["next_close"]),
        row_index=int(path["holding_days"]),
        exit_date=pd.Timestamp(last["execution_date"]).date(),
        reason="horizon_exit",
        outcome="HOLD_TO_HORIZON",
        config=config,
        apply_slippage=True,
    )
    hold_return = float(state["cash"] - 1.0)
    return _finalize_result(
        path_id=path_id,
        path=path,
        strategy=HOLD_STRATEGY,
        state=state,
        hold_return=hold_return,
        config=config,
    )


def _simulate_risk_stop(
    path_id: int,
    path: dict[str, Any],
    config: AppConfig,
    *,
    strategy: str,
    atr_multiplier: float,
    minimum_stop_gap_pct: float,
    hold_return: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = _base_state(path, config)
    state["initial_execution_price"] = float(state["cost"])
    active_stop: float | None = None
    for index, row in path["rows"].iterrows():
        state["time_in_market"] += 1
        close = float(row["raw_close"])
        recommendation = replace(
            config.recommendation,
            atr_stop_multiplier=float(atr_multiplier),
            minimum_stop_gap_pct=float(minimum_stop_gap_pct),
            stop_limit_slippage_pct=float(config.chart_exit.limit_slippage_pct),
        )
        decision = make_risk_decision(
            close=close,
            avg_cost=float(state["cost"]),
            max_loss_pct=float(config.positions.default_max_loss_pct),
            current_stop=active_stop,
            predicted_high_return=float(row["predicted_high_return"]),
            predicted_low_return=float(row["predicted_low_return"]),
            similar_high_return=float(row["median_high_return"]),
            similar_low_return=float(row["low_return_q20"]),
            atr_14=_atr(row, close),
            similarity_distance=float(row["mean_distance"]),
            sample_count=int(row["sample_count"]),
            config=recommendation,
        )
        active_stop = (
            float(decision.stop_trigger_price)
            if active_stop is None
            else max(active_stop, float(decision.stop_trigger_price))
        )
        active_stop = min(active_stop, close)
        stop_limit = round_to_tick(
            active_stop * (1.0 - float(config.chart_exit.limit_slippage_pct)),
            float(config.recommendation.price_tick),
            "down",
        )
        simulation = simulate_next_day(
            close_price=close,
            take_profit_price=None,
            stop_trigger_price=active_stop,
            stop_limit_price=stop_limit,
            next_open=float(row["next_open"]),
            next_high=float(row["next_high"]),
            next_low=float(row["next_low"]),
            next_close=float(row["next_close"]),
            next_is_suspended=_is_next_suspended(row),
            stop_order_type=config.chart_exit.stop_order_type,
        )
        outcome = str(simulation["outcome"])
        if outcome == "STOP_LIMIT_UNFILLED":
            state["unfilled_count"] += 1
        if outcome not in NO_EXIT_OUTCOMES:
            _sell_position(
                state,
                quote=float(simulation["exit_price"]),
                row_index=int(index) + 1,
                exit_date=pd.Timestamp(row["execution_date"]).date(),
                reason="risk_stop",
                outcome=outcome,
                config=config,
                apply_slippage=False,
            )
            break
        mark = _portfolio_mark(state["shares"], 0.0, float(row["next_close"]))
        adverse = _portfolio_mark(state["shares"], 0.0, float(row["next_low"]))
        state["equity"].append(mark)
        state["adverse"].append(adverse - 1.0)
        state["max_equity"] = max(
            float(state["max_equity"]),
            mark,
            _portfolio_mark(state["shares"], 0.0, float(row["next_high"])),
        )

    if state["shares"] > 0:
        last = path["rows"].iloc[-1]
        _sell_position(
            state,
            quote=float(last["next_close"]),
            row_index=int(path["holding_days"]),
            exit_date=pd.Timestamp(last["execution_date"]).date(),
            reason="horizon_exit",
            outcome="HOLD_TO_HORIZON",
            config=config,
            apply_slippage=True,
        )
    return _finalize_result(
        path_id=path_id,
        path=path,
        strategy=strategy,
        state=state,
        hold_return=hold_return,
        config=config,
    )


def _new_trade_chart_state(state: dict[str, Any], row: pd.Series, config: AppConfig) -> dict[str, Any]:
    cost = float(state["cost"])
    stop = _initial_stop(cost, row, config)
    risk_r = max(
        cost - stop,
        float(config.chart_exit.r_floor_atr_mult) * _atr(row, cost),
        float(config.recommendation.price_tick),
    )
    hard_stop = cost * (1.0 - float(config.positions.default_max_loss_pct))
    return {
        "initial_stop": stop,
        "disaster_stop": round_to_tick(
            hard_stop,
            float(config.recommendation.price_tick),
            "down",
        ),
        "risk_r": risk_r,
        "progress_price": cost + float(config.chart_exit.progress_r) * risk_r,
        "held_peak_close": cost,
        "entry_swing_low": float(row["recent_swing_low"]),
        "trend": False,
        "below_ma_count": 0,
        "below_fast_count": 0,
    }


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _row_price(row: pd.Series, raw_column: str, adjusted_column: str) -> float | None:
    if raw_column in row and pd.notna(row.get(raw_column)):
        return _finite_float(row.get(raw_column))
    return _finite_float(row.get(adjusted_column))


def _with_position_max_loss(config: AppConfig, max_loss: float) -> AppConfig:
    positions = replace(config.positions, default_max_loss_pct=float(max_loss))
    if is_dataclass(config):
        return replace(config, positions=positions)
    return SimpleNamespace(
        chart_exit=config.chart_exit,
        positions=positions,
        recommendation=config.recommendation,
    )


def recommend_chart_exit(
    history: pd.DataFrame,
    position: pd.Series | dict[str, Any],
    as_of_date: date,
    config: AppConfig,
) -> ChartExitRecommendation:
    settings = config.chart_exit
    profile = str(settings.production_profile)
    position_scale = float(settings.position_scale)
    tick = float(config.recommendation.price_tick)
    avg_cost = _finite_float(position.get("avg_cost"))
    max_loss = _finite_float(position.get("max_loss_pct"))
    if max_loss is None or not 0 < max_loss < 1:
        max_loss = float(config.positions.default_max_loss_pct)

    current_close: float | None = None

    def unavailable(reason: str, diagnostic: str) -> ChartExitRecommendation:
        trigger = np.nan
        limit = np.nan
        if avg_cost is not None:
            raw_trigger = avg_cost * (1.0 - max_loss)
            if current_close is not None:
                raw_trigger = min(raw_trigger, current_close)
            trigger = round_to_tick(raw_trigger, tick, "down")
            limit = round_to_tick(
                trigger * (1.0 - float(settings.limit_slippage_pct)),
                tick,
                "down",
            )
        return ChartExitRecommendation(
            action="HOLD",
            reason=reason,
            stop_trigger_price=float(trigger),
            stop_limit_price=float(limit),
            profile=profile,
            position_scale=position_scale,
            diagnostic=diagnostic,
        )

    if avg_cost is None or avg_cost <= 0:
        return unavailable("chart_exit_unavailable", "avg_cost_missing")
    if history.empty:
        return unavailable("chart_exit_unavailable", "history_empty")

    frame = history.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    as_of_timestamp = pd.Timestamp(as_of_date).normalize()
    frame = frame.loc[frame["trade_date"].dt.normalize() <= as_of_timestamp].reset_index(drop=True)
    if frame.empty:
        return unavailable("chart_exit_unavailable", "history_before_as_of_empty")

    current_matches = frame.index[
        frame["trade_date"].dt.normalize() == as_of_timestamp
    ].tolist()
    if not current_matches:
        return unavailable("chart_exit_unavailable", "as_of_bar_missing")
    current_index = int(current_matches[-1])
    current_row = frame.iloc[current_index]
    current_close = _row_price(current_row, "raw_close", "close")
    if current_close is None:
        return unavailable("chart_exit_unavailable", "current_close_missing")

    diagnostics: list[str] = []
    entry_value = position.get("entry_date")
    entry_timestamp = pd.to_datetime(entry_value, errors="coerce")
    if pd.isna(entry_timestamp):
        entry_index = 0
        diagnostics.append("entry_date_missing")
    else:
        entry_timestamp = pd.Timestamp(entry_timestamp).normalize()
        if entry_timestamp > as_of_timestamp:
            return unavailable("chart_exit_unavailable", "entry_date_after_as_of")
        first_available = pd.Timestamp(frame.iloc[0]["trade_date"]).normalize()
        if entry_timestamp < first_available:
            diagnostics.append("entry_history_truncated")
        entry_matches = frame.index[
            (frame["trade_date"].dt.normalize() >= entry_timestamp)
            & (frame.index <= current_index)
        ].tolist()
        entry_index = int(entry_matches[0]) if entry_matches else 0

    entry_row = frame.iloc[entry_index]
    for column in ("recent_swing_low", "atr_14_pct"):
        if _finite_float(entry_row.get(column)) is None:
            return unavailable("chart_exit_unavailable", f"entry_{column}_missing")

    local_config = _with_position_max_loss(config, max_loss)
    chart = _new_trade_chart_state({"cost": avg_cost}, entry_row, local_config)
    current_reason: str | None = None
    current_trade_days = 0
    window = frame.iloc[entry_index : current_index + 1].reset_index(drop=True)
    for offset, row in window.iterrows():
        close = _row_price(row, "raw_close", "close")
        ma_fast = _finite_float(row.get("ma_fast"))
        ma_trend = _finite_float(row.get("ma_trend"))
        ma_long = _finite_float(row.get("ma_long"))
        ma_slope = _finite_float(row.get("ma_trend_slope"))
        recent_swing_low = _finite_float(row.get("recent_swing_low"))
        if None in (close, ma_fast, ma_trend, ma_long, ma_slope, recent_swing_low):
            if int(offset) == len(window) - 1:
                return unavailable("chart_exit_unavailable", "current_indicator_missing")
            diagnostics.append("indicator_gap")
            continue

        current_trade_days = int(offset) + 1
        chart["held_peak_close"] = max(float(chart["held_peak_close"]), float(close))
        if (
            not chart["trend"]
            and float(close) >= float(chart["progress_price"])
            and float(close) > float(ma_trend)
            and float(ma_slope) > 0
        ):
            chart["trend"] = True

        floor = _profit_floor(avg_cost, float(chart["held_peak_close"]), local_config)
        next_open_reason: str | None = None
        if np.isfinite(floor) and float(close) < floor:
            next_open_reason = "profit_floor_confirmed"
        if chart["trend"]:
            if float(close) < float(ma_trend):
                chart["below_ma_count"] += 1
            else:
                chart["below_ma_count"] = 0
            if next_open_reason is None and float(close) < float(ma_long):
                next_open_reason = "ma_long_breakdown"
            elif (
                next_open_reason is None
                and chart["below_ma_count"] >= int(settings.breakdown_confirm_closes)
                and float(close) < float(recent_swing_low)
            ):
                next_open_reason = "ma_and_swing_confirmed"
        else:
            if float(close) < float(ma_fast):
                chart["below_fast_count"] += 1
            else:
                chart["below_fast_count"] = 0
            if next_open_reason is None and (
                chart["below_fast_count"] >= int(settings.breakdown_confirm_closes)
                and float(close) < float(chart["entry_swing_low"])
            ):
                next_open_reason = "initial_structure_breakdown"
            elif next_open_reason is None and (
                current_trade_days >= int(settings.initial_days)
                and float(chart["held_peak_close"]) < float(chart["progress_price"])
                and float(close) < avg_cost
                and float(close) < float(ma_fast)
                and chart["below_fast_count"] >= int(settings.breakdown_confirm_closes)
            ):
                next_open_reason = "initial_failure"

        if int(offset) == len(window) - 1:
            current_reason = next_open_reason

    disaster_trigger = min(float(chart["disaster_stop"]), float(current_close))
    disaster_trigger = round_to_tick(disaster_trigger, tick, "down")
    stop_limit = round_to_tick(
        disaster_trigger * (1.0 - float(settings.limit_slippage_pct)),
        tick,
        "down",
    )
    return ChartExitRecommendation(
        action="EXIT_NEXT_OPEN" if current_reason is not None else "HOLD",
        reason=current_reason or "chart_hold",
        stop_trigger_price=float(disaster_trigger),
        stop_limit_price=float(stop_limit),
        profile=profile,
        position_scale=position_scale,
        held_peak_close=float(chart["held_peak_close"]),
        trend_active=bool(chart["trend"]),
        trade_days=int(current_trade_days),
        diagnostic=";".join(dict.fromkeys(diagnostics)),
    )


def _reentry_signal(
    row: pd.Series,
    *,
    state: dict[str, Any],
    last_exit: dict[str, Any],
    current_index: int,
    config: AppConfig,
) -> bool:
    settings = config.chart_exit
    if state["entries"] >= int(settings.max_entries_safety):
        return False
    if current_index - int(last_exit["exit_day_index"]) < int(settings.reentry_cooldown_days):
        return False
    close = float(row["raw_close"])
    ma_trend = float(row["ma_trend"])
    ma_long = float(row["ma_long"])
    prior_high = float(row["prior_breakout_high"])
    breakout = max(float(last_exit["exit_candle_high"]), prior_high)
    common = close > ma_trend and float(row["ma_trend_slope"]) > 0 and close > breakout
    if not common:
        return False
    if last_exit["exit_reason"] in {
        "hard_stop",
        "initial_structure_breakdown",
        "initial_failure",
    }:
        return bool(ma_trend > ma_long and close > ma_long)
    return True


def _simulate_chart(
    path_id: int,
    path: dict[str, Any],
    config: AppConfig,
    *,
    allow_reentry: bool,
    hold_return: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    strategy = CHART_REENTRY_STRATEGY if allow_reentry else CHART_STRATEGY
    state = _base_state(path, config)
    state["initial_execution_price"] = float(state["cost"])
    chart = _new_trade_chart_state(state, path["rows"].iloc[0], config)
    last_exit: dict[str, Any] | None = None
    tick = float(config.recommendation.price_tick)

    for index, row in path["rows"].iterrows():
        execution_date = pd.Timestamp(row["execution_date"]).date()
        if state["shares"] <= 0:
            state["equity"].append(float(state["cash"]))
            state["adverse"].append(float(state["cash"]) - 1.0)
            if (
                allow_reentry
                and last_exit is not None
                and not _is_next_suspended(row)
                and _reentry_signal(
                    row,
                    state=state,
                    last_exit=last_exit,
                    current_index=int(index),
                    config=config,
                )
            ):
                quote = float(row["next_open"])
                candidate_stop = _initial_stop(quote, row, config)
                risk_pct = max(0.0, quote - candidate_stop) / quote
                if risk_pct <= float(config.chart_exit.max_reentry_risk_pct):
                    _enter_position(
                        state,
                        quote=quote,
                        entry_date=execution_date,
                        entry_index=int(index) + 1,
                        config=config,
                    )
                    chart = _new_trade_chart_state(state, row, config)
                else:
                    continue
            else:
                continue

        state["time_in_market"] += 1
        close = float(row["raw_close"])
        cost = float(state["cost"])
        chart["held_peak_close"] = max(float(chart["held_peak_close"]), close)

        if (
            not chart["trend"]
            and close >= float(chart["progress_price"])
            and close > float(row["ma_trend"])
            and float(row["ma_trend_slope"]) > 0
        ):
            chart["trend"] = True

        # Ordinary exits are close-confirmed in v2. Only the fixed 5% disaster
        # line remains an intraday order, so normal lower shadows cannot force
        # an exit by themselves.
        floor = _profit_floor(cost, float(chart["held_peak_close"]), config)

        next_open_reason: str | None = None
        if np.isfinite(floor) and close < floor:
            next_open_reason = "profit_floor_confirmed"
        if chart["trend"]:
            if close < float(row["ma_trend"]):
                chart["below_ma_count"] += 1
            else:
                chart["below_ma_count"] = 0
            if next_open_reason is None and close < float(row["ma_long"]):
                next_open_reason = "ma_long_breakdown"
            elif (
                next_open_reason is None
                and chart["below_ma_count"]
                >= int(config.chart_exit.breakdown_confirm_closes)
                and close < float(row["recent_swing_low"])
            ):
                next_open_reason = "ma_and_swing_confirmed"
        else:
            if close < float(row["ma_fast"]):
                chart["below_fast_count"] += 1
            else:
                chart["below_fast_count"] = 0
            trade_days = int(index) - int(state["entry_index"]) + 1
            if next_open_reason is None and (
                chart["below_fast_count"]
                >= int(config.chart_exit.breakdown_confirm_closes)
                and close < float(chart["entry_swing_low"])
            ):
                next_open_reason = "initial_structure_breakdown"
            elif next_open_reason is None and (
                trade_days >= int(config.chart_exit.initial_days)
                and float(chart["held_peak_close"]) < float(chart["progress_price"])
                and close < cost
                and close < float(row["ma_fast"])
                and chart["below_fast_count"]
                >= int(config.chart_exit.breakdown_confirm_closes)
            ):
                next_open_reason = "initial_failure"

        if next_open_reason is not None and not _is_next_suspended(row):
            _sell_position(
                state,
                quote=float(row["next_open"]),
                row_index=int(index) + 1,
                exit_date=execution_date,
                reason=next_open_reason,
                outcome="NEXT_OPEN_EXIT",
                config=config,
                apply_slippage=True,
            )
            last_trade = state["trades"][-1]
            last_exit = {
                **last_trade,
                "exit_candle_high": float(row["next_high"]),
            }
            continue

        disaster_trigger = min(float(chart["disaster_stop"]), close)
        disaster_trigger = round_to_tick(disaster_trigger, tick, "down")
        stop_limit = round_to_tick(
            disaster_trigger
            * (1.0 - float(config.chart_exit.limit_slippage_pct)),
            tick,
            "down",
        )
        simulation = simulate_next_day(
            close_price=close,
            take_profit_price=None,
            stop_trigger_price=disaster_trigger,
            stop_limit_price=stop_limit,
            next_open=float(row["next_open"]),
            next_high=float(row["next_high"]),
            next_low=float(row["next_low"]),
            next_close=float(row["next_close"]),
            next_is_suspended=_is_next_suspended(row),
            stop_order_type=config.chart_exit.stop_order_type,
        )
        outcome = str(simulation["outcome"])
        if outcome == "STOP_LIMIT_UNFILLED":
            state["unfilled_count"] += 1
        if outcome not in NO_EXIT_OUTCOMES:
            _sell_position(
                state,
                quote=float(simulation["exit_price"]),
                row_index=int(index) + 1,
                exit_date=execution_date,
                reason="hard_stop",
                outcome=outcome,
                config=config,
                apply_slippage=False,
            )
            last_trade = state["trades"][-1]
            last_exit = {
                **last_trade,
                "exit_candle_high": float(row["next_high"]),
            }
            continue

        mark = _portfolio_mark(state["shares"], 0.0, float(row["next_close"]))
        adverse = _portfolio_mark(state["shares"], 0.0, float(row["next_low"]))
        state["equity"].append(mark)
        state["adverse"].append(adverse - 1.0)
        state["max_equity"] = max(
            float(state["max_equity"]),
            mark,
            _portfolio_mark(state["shares"], 0.0, float(row["next_high"])),
        )

    if state["shares"] > 0:
        last = path["rows"].iloc[-1]
        _sell_position(
            state,
            quote=float(last["next_close"]),
            row_index=int(path["holding_days"]),
            exit_date=pd.Timestamp(last["execution_date"]).date(),
            reason="horizon_exit",
            outcome="HOLD_TO_HORIZON",
            config=config,
            apply_slippage=True,
        )
    return _finalize_result(
        path_id=path_id,
        path=path,
        strategy=strategy,
        state=state,
        hold_return=hold_return,
        config=config,
    )


def _simulate_one_path(
    path_id: int,
    path: dict[str, Any],
    config: AppConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hold_result, hold_ledger = _simulate_hold(path_id, path, config)
    hold_return = float(hold_result["strategy_return"])
    path_results = [(hold_result, hold_ledger)]
    path_results.append(
        _simulate_risk_stop(
            path_id,
            path,
            config,
            strategy=PURE_2PCT_STRATEGY,
            atr_multiplier=0.0,
            minimum_stop_gap_pct=0.02,
            hold_return=hold_return,
        )
    )
    path_results.append(
        _simulate_risk_stop(
            path_id,
            path,
            config,
            strategy=PRODUCTION_STRATEGY,
            atr_multiplier=float(config.recommendation.atr_stop_multiplier),
            minimum_stop_gap_pct=float(config.recommendation.minimum_stop_gap_pct),
            hold_return=hold_return,
        )
    )
    path_results.append(
        _simulate_chart(
            path_id,
            path,
            config,
            allow_reentry=False,
            hold_return=hold_return,
        )
    )
    path_results.append(
        _simulate_chart(
            path_id,
            path,
            config,
            allow_reentry=True,
            hold_return=hold_return,
        )
    )
    records: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for result, trades in path_results:
        records.append(result)
        ledger.extend(trades)
    return records, ledger


def _simulate_paths(
    paths: list[dict[str, Any]],
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    workers = max(1, int(config.performance.workers))
    items = list(enumerate(paths))

    def run(item: tuple[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return _simulate_one_path(item[0], item[1], config)

    iterator = map(run, items)
    executor: ThreadPoolExecutor | None = None
    if workers > 1:
        executor = ThreadPoolExecutor(max_workers=workers)
        iterator = executor.map(run, items)
    try:
        for completed, (path_records, path_ledger) in enumerate(iterator, start=1):
            records.extend(path_records)
            ledger.extend(path_ledger)
            if completed % 100 == 0 or completed == len(paths):
                print(
                    f"K线持仓模拟进度 {completed}/{len(paths)}（{workers} 线程）",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return pd.DataFrame(records), pd.DataFrame(ledger)


def summarize_chart_results(frame: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("没有可汇总的K线持仓回测结果")
    summary: dict[str, Any] = {
        "path_count": int(len(frame)),
        "average_net_return": float(frame["strategy_return"].mean()),
        "median_net_return": float(frame["strategy_return"].median()),
        "p05_net_return": float(frame["strategy_return"].quantile(0.05)),
        "average_hold_return": float(frame["hold_return"].mean()),
        "average_excess_vs_hold": float(frame["excess_return"].mean()),
        "average_max_drawdown": float(frame["strategy_max_drawdown"].mean()),
        "average_max_loss": float(frame["strategy_max_loss"].mean()),
        "win_rate": float(frame["win"].mean()),
        "stopped_rate": float(frame["stopped"].mean()),
        "whipsaw_rate": float(frame["whipsaw"].mean()),
        "trend_retention_rate": float(frame["trend_retained"].mean()),
        "profit_giveback_mean": float(frame["profit_giveback"].mean()),
        "average_profit_giveback_pct_points": float(
            frame["profit_giveback_pct_points"].mean()
        ),
        "average_time_in_market_days": float(frame["time_in_market_days"].mean()),
        "average_entry_count": float(frame["entry_count"].mean()),
        "average_reentry_count": float(frame["reentry_count"].mean()),
        "reentry_success_rate": (
            float(frame["reentry_success_count"].sum() / frame["reentry_count"].sum())
            if frame["reentry_count"].sum() > 0
            else 0.0
        ),
        "average_turnover": float(frame["turnover"].mean()),
        "average_transaction_cost_rate": float(frame["transaction_cost_rate"].mean()),
        "average_max_consecutive_failed_entries": float(
            frame["max_consecutive_failed_entries"].mean()
        ),
        "stop_limit_unfilled_rate": float(
            (frame["stop_limit_unfilled_count"] > 0).mean()
        ),
    }
    winners = frame["full_path_peak_gain"] >= float(config.chart_exit.winner_peak_gain)
    if winners.any():
        summary["trend_retention_rate"] = float(frame.loc[winners, "trend_retained"].mean())
        capture = frame.loc[winners, "strategy_return"] / frame.loc[
            winners, "full_path_peak_gain"
        ].clip(lower=1e-9)
        summary["average_trend_capture_ratio"] = float(capture.mean())
    else:
        summary["average_trend_capture_ratio"] = 0.0
    for days in config.chart_exit.sell_fly_days:
        for threshold in config.chart_exit.sell_fly_thresholds:
            label = f"sell_fly_{int(days)}d_{int(round(threshold * 100))}pct"
            observable = frame[f"{label}_observable"].astype(bool)
            summary[f"{label}_all_paths"] = float(frame[label].mean())
            summary[f"{label}_observable"] = (
                float(frame.loc[observable, label].mean()) if observable.any() else 0.0
            )
    return summary


def _summaries(frame: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    return {
        strategy: summarize_chart_results(group, config)
        for strategy, group in frame.groupby("strategy", sort=True)
    }


def _candidate_diagnostics(
    candidate: dict[str, Any],
    production: dict[str, Any],
    config: AppConfig,
) -> dict[str, Any]:
    metrics = (
        "average_net_return",
        "p05_net_return",
        "average_excess_vs_hold",
        "average_max_drawdown",
        "average_max_loss",
        "whipsaw_rate",
        "trend_retention_rate",
        "average_trend_capture_ratio",
        "profit_giveback_mean",
        "average_time_in_market_days",
        "average_reentry_count",
        "average_turnover",
        "average_transaction_cost_rate",
    )
    deltas = {metric: float(candidate[metric]) - float(production[metric]) for metric in metrics}
    tolerance = float(config.chart_exit.hard_constraint_tolerance_pct)
    hard = {
        "p05_not_worse_0_5pp": deltas["p05_net_return"] >= -tolerance,
        "max_loss_not_worse_0_5pp": deltas["average_max_loss"] >= -tolerance,
        "max_drawdown_not_worse_0_5pp": deltas["average_max_drawdown"] >= -tolerance,
    }
    primary = {
        "net_return_higher": deltas["average_net_return"] > 0,
        "excess_vs_hold_higher": deltas["average_excess_vs_hold"] > 0,
        "trend_retention_higher": deltas["trend_retention_rate"] > 0,
        "profit_giveback_lower": deltas["profit_giveback_mean"] < 0,
    }
    return {
        "pass": bool(all(hard.values()) and sum(primary.values()) >= 3),
        "hard_constraints": hard,
        "primary_edges": primary,
        "primary_edge_count": int(sum(primary.values())),
        "deltas_candidate_minus_production": deltas,
    }


def _diagnostics(summaries: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    production = summaries[PRODUCTION_STRATEGY]
    candidates = {
        strategy: _candidate_diagnostics(summaries[strategy], production, config)
        for strategy in (CHART_STRATEGY, CHART_REENTRY_STRATEGY)
    }
    d_minus_c = {
        metric: float(summaries[CHART_REENTRY_STRATEGY][metric])
        - float(summaries[CHART_STRATEGY][metric])
        for metric in (
            "average_net_return",
            "p05_net_return",
            "average_max_drawdown",
            "average_max_loss",
            "trend_retention_rate",
            "profit_giveback_mean",
            "average_reentry_count",
            "average_turnover",
        )
    }
    return {
        "pass": bool(any(value["pass"] for value in candidates.values())),
        "candidates_vs_production": candidates,
        "reentry_increment_d_minus_c": d_minus_c,
    }


def _entry_months(samples: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(samples["entry_date"]).dt.to_period("M").astype(str)


def _walk_forward_windows(samples: pd.DataFrame, config: AppConfig) -> list[dict[str, Any]]:
    months = sorted(_entry_months(samples).unique())
    settings = config.chart_exit
    train = int(settings.walk_forward_train_months)
    validation = int(settings.validation_months)
    step = int(settings.walk_forward_step_months)
    if len(months) < train + validation:
        return []
    entry_months = _entry_months(samples)
    windows: list[dict[str, Any]] = []
    for no, validation_start in enumerate(
        range(train, len(months) - validation + 1, step),
        start=1,
    ):
        train_values = months[validation_start - train : validation_start]
        validation_values = months[validation_start : validation_start + validation]
        windows.append(
            {
                "window_id": f"wf_{no:02d}",
                "train_months": train_values,
                "validation_months": validation_values,
                "validation_path_ids": set(
                    samples.loc[entry_months.isin(validation_values), "path_id"]
                    .astype(int)
                    .tolist()
                ),
            }
        )
    return windows


def _bootstrap(
    frame: pd.DataFrame,
    config: AppConfig,
    *,
    candidate: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0:
        return {"status": "skipped", "reason": "bootstrap disabled"}
    path_ids = frame["path_id"].astype(int).drop_duplicates().to_numpy()
    if len(path_ids) < 2:
        return {"status": "skipped", "reason": "not enough paths"}
    rng = np.random.default_rng(seed)
    by_path = {
        int(path_id): frame.index[frame["path_id"].astype(int) == int(path_id)].to_numpy()
        for path_id in path_ids
    }
    collected: dict[str, list[float]] = {}
    pass_count = 0
    for _ in range(iterations):
        sampled = rng.choice(path_ids, size=len(path_ids), replace=True)
        indices = np.concatenate([by_path[int(path_id)] for path_id in sampled])
        summaries = _summaries(frame.loc[indices], config)
        if candidate not in summaries or PRODUCTION_STRATEGY not in summaries:
            continue
        diagnostic = _candidate_diagnostics(
            summaries[candidate], summaries[PRODUCTION_STRATEGY], config
        )
        pass_count += int(diagnostic["pass"])
        for metric, value in diagnostic["deltas_candidate_minus_production"].items():
            collected.setdefault(metric, []).append(float(value))
    return {
        "status": "ok",
        "iterations": int(iterations),
        "path_count": int(len(path_ids)),
        "pass_rate": float(pass_count / iterations),
        "deltas": {
            metric: {
                "mean": float(np.mean(values)),
                "standard_error": float(np.std(values, ddof=1)),
                "p05": float(np.quantile(values, 0.05)),
                "p95": float(np.quantile(values, 0.95)),
            }
            for metric, values in collected.items()
        },
    }


def _aggregate_walk_forward(records: list[dict[str, Any]], config: AppConfig) -> dict[str, Any]:
    ok = [record for record in records if record.get("status") == "ok"]
    if not ok:
        return {"status": "skipped", "reason": "no successful windows"}
    aggregate: dict[str, Any] = {}
    for candidate in (CHART_STRATEGY, CHART_REENTRY_STRATEGY):
        candidate_records = [
            record["diagnostics"]["candidates_vs_production"][candidate] for record in ok
        ]
        passes = [bool(value["pass"]) for value in candidate_records]
        delta_frame = pd.DataFrame(
            [value["deltas_candidate_minus_production"] for value in candidate_records]
        )
        delta_summary: dict[str, Any] = {}
        for metric in delta_frame.columns:
            values = delta_frame[metric].astype(float)
            delta_summary[metric] = {
                "mean": float(values.mean()),
                "standard_error_across_windows": (
                    float(values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0
                ),
                "positive_window_rate": float((values > 0).mean()),
            }
        pass_rate = float(np.mean(passes))
        average_return = delta_summary["average_net_return"]
        tolerance = float(config.chart_exit.hard_constraint_tolerance_pct)
        hard = (
            delta_summary["p05_net_return"]["mean"] >= -tolerance
            and delta_summary["average_max_loss"]["mean"] >= -tolerance
            and delta_summary["average_max_drawdown"]["mean"] >= -tolerance
        )
        stable_return_edge = (
            average_return["mean"] > average_return["standard_error_across_windows"]
        )
        aggregate[candidate] = {
            "window_count": int(len(ok)),
            "pass_rate": pass_rate,
            "pass": bool(hard and pass_rate >= 0.5 and stable_return_edge),
            "hard_constraints_pass": bool(hard),
            "stable_net_return_edge": bool(stable_return_edge),
            "delta_summary": delta_summary,
        }
    return {
        "status": "ok",
        "candidates": aggregate,
        "pass": bool(any(value["pass"] for value in aggregate.values())),
    }


def _run_walk_forward(
    samples: pd.DataFrame,
    results: pd.DataFrame,
    config: AppConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    windows = _walk_forward_windows(samples, config)
    if not windows:
        return (
            {"status": "skipped", "reason": "not enough entry months"},
            pd.DataFrame(),
        )
    records: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for window in windows:
        frame = results.loc[
            results["path_id"].astype(int).isin(window["validation_path_ids"])
        ].copy()
        if frame.empty:
            continue
        summaries = _summaries(frame, config)
        diagnostics = _diagnostics(summaries, config)
        records.append(
            {
                "window_id": window["window_id"],
                "status": "ok",
                "train_months": window["train_months"],
                "validation_months": window["validation_months"],
                "validation_path_count": int(len(window["validation_path_ids"])),
                "summaries": summaries,
                "diagnostics": diagnostics,
                "bootstrap": {
                    candidate: _bootstrap(
                        frame,
                        config,
                        candidate=candidate,
                        iterations=int(config.chart_exit.bootstrap_samples),
                        seed=int(config.training.random_seed),
                    )
                    for candidate in (CHART_STRATEGY, CHART_REENTRY_STRATEGY)
                },
            }
        )
        frame["window_id"] = window["window_id"]
        frames.append(frame)
    aggregate = _aggregate_walk_forward(records, config)
    return (
        {
            "status": "ok" if records else "failed",
            "records": records,
            "aggregate": aggregate,
            "pass": bool(aggregate.get("pass", False)),
        },
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
    )


def run_chart_exit_backtest(
    engine: Any,
    config: AppConfig,
    *,
    start_date: date,
    end_date: date,
    max_symbols: int | None = None,
    output_directory: str | Path = "output/chart-exit",
) -> dict[str, Any]:
    if start_date > end_date:
        raise ValueError("K线持仓回测开始日期不能晚于结束日期")
    settings = config.chart_exit
    max_symbols = max_symbols or int(settings.max_symbols)
    all_entry_dates = _entry_dates(engine, start_date, end_date)
    if not all_entry_dates:
        raise RuntimeError("K线持仓回测范围没有交易日")
    max_rows = max(settings.holding_days) + int(settings.lookahead_extend_days)
    entry_dates: list[date] = []
    skipped: list[date] = []
    for entry_date in all_entry_dates:
        try:
            _path_end_date(engine, entry_date, max_rows)
            entry_dates.append(entry_date)
        except RuntimeError:
            skipped.append(entry_date)
    if not entry_dates:
        raise RuntimeError("没有满足持有期和观察延伸期的完整建仓月份")
    path_end_date = _path_end_date(engine, entry_dates[-1], max_rows)
    symbols = _select_evaluation_symbols(engine, config, start_date, max_symbols)
    if not symbols:
        raise RuntimeError("没有可用于K线持仓回测的股票")

    features = _prepare_chart_features(
        engine,
        config,
        entry_dates[0],
        path_end_date,
        symbols,
    )
    features = features.loc[
        (features["trade_date"].dt.date >= entry_dates[0])
        & (features["trade_date"].dt.date < path_end_date)
    ].copy()
    predicted = _predict_stops(engine, config, features)
    if predicted.empty:
        raise RuntimeError("K线持仓回测没有可用的模型/相似形态特征")
    paths = _build_paths(
        predicted,
        symbols,
        entry_dates,
        settings.holding_days,
        lookahead_days=int(settings.lookahead_extend_days),
    )
    if not paths:
        raise RuntimeError("没有构建出完整的K线持仓回测路径")
    print(
        f"K线持仓回测：{len(symbols)} 支股票，{len(entry_dates)} 个建仓月，"
        f"{len(paths)} 条路径，5 个策略",
        flush=True,
    )

    results, ledger = _simulate_paths(paths, config)
    samples = _path_records(paths)
    summaries = _summaries(results, config)
    diagnostics = _diagnostics(summaries, config)
    walk_forward, walk_forward_paths = _run_walk_forward(samples, results, config)

    run_id = datetime.now(timezone.utc).strftime("ce_%Y%m%dT%H%M%SZ")
    output_root = Path(output_directory).resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    paths_path = output_root / "paths.csv"
    results_path = output_root / "chart_paths.csv"
    ledger_path = output_root / "trade_ledger.csv"
    walk_forward_path = output_root / "walk_forward_summary.json"
    walk_forward_paths_path = output_root / "walk_forward_paths.csv"
    summary_path = output_root / "summary.json"
    samples.to_csv(paths_path, index=False, encoding="utf-8-sig")
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    ledger.to_csv(ledger_path, index=False, encoding="utf-8-sig")
    if not walk_forward_paths.empty:
        walk_forward_paths.to_csv(
            walk_forward_paths_path,
            index=False,
            encoding="utf-8-sig",
        )
    walk_forward_path.write_text(
        json.dumps(walk_forward, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    summary = {
        "run_id": run_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "path_end_date": path_end_date.isoformat(),
        "symbol_count": int(len(symbols)),
        "entry_dates": [value.isoformat() for value in entry_dates],
        "skipped_incomplete_entry_dates": [value.isoformat() for value in skipped],
        "path_count": int(len(paths)),
        "strategy_count": int(len(STRATEGIES)),
        "summaries": summaries,
        "diagnostics": diagnostics,
        "walk_forward": walk_forward,
        "chart_exit_config": asdict(settings),
        "promotion_action": "report_only",
        "production_changed": False,
        "output_directory": str(output_root),
        "paths_path": str(paths_path),
        "chart_paths_path": str(results_path),
        "trade_ledger_path": str(ledger_path),
        "walk_forward_summary_path": str(walk_forward_path),
        "walk_forward_paths_path": (
            str(walk_forward_paths_path) if not walk_forward_paths.empty else None
        ),
        "method": (
            "同批月度建仓路径配对比较：H=持有，A=纯2%跟踪，B=当前生产ATR/模型止损，"
            "C=盘中5%防灾+收盘结构/均线确认+最高收盘价分段利润底，"
            "D=C加条件式可重复再上车。所有收益均扣除佣金、卖出印花税和配置滑点；"
            "参数为固定结构v2，不做网格寻优。"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary
