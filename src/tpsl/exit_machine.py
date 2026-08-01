from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import _select_evaluation_symbols, simulate_next_day
from .config import AppConfig
from .db import load_bars
from .features import build_feature_frame
from .holding_backtest import _maximum_drawdown
from .risk import round_to_tick
from .stop_tuning import _entry_dates, _path_end_date


STATE_MACHINE_STRATEGY = "exit_machine"
FIXED_BASELINE_STRATEGY = "fixed_2pct"


def _feature_start(entry_date: date, config: AppConfig) -> date:
    lookback = max(
        300,
        int(config.exit_machine.breakdown_ma) + 260,
        int(config.exit_machine.ma_fast) + 260,
    )
    return entry_date - timedelta(days=lookback)


def _recent_confirmed_swing_lows(lows: pd.Series, *, pivot_k: int) -> pd.Series:
    values = pd.to_numeric(lows, errors="coerce").to_numpy(dtype=float)
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
                and np.all(value < left)
                and np.all(value < right)
            ):
                last_confirmed = value
        recent[current] = last_confirmed
    return pd.Series(recent, index=lows.index)


def _prepare_exit_features(
    engine: Any,
    config: AppConfig,
    earliest_entry: date,
    load_end_date: date,
    symbols: list[str],
) -> pd.DataFrame:
    bars = load_bars(
        engine,
        config,
        _feature_start(earliest_entry, config),
        load_end_date,
        symbols=symbols,
    )
    if bars.empty:
        raise RuntimeError("状态机退出回测范围没有行情数据")
    print(f"状态机退出特征行情读取完成：{len(bars):,} 行", flush=True)
    features = build_feature_frame(
        bars,
        workers=config.performance.workers,
        min_symbol_rows=25,
    )
    if features.empty:
        raise RuntimeError("状态机退出特征为空")

    pivot_k = int(config.exit_machine.swing_pivot_k)
    parts: list[pd.DataFrame] = []
    for _, group in features.groupby("symbol", sort=False):
        group = group.sort_values("trade_date").copy()
        close = group["raw_close"].astype(float)
        group["ma_10"] = close.rolling(10, min_periods=1).mean()
        group["ma_20"] = close.rolling(20, min_periods=1).mean()
        group["rolling_low_10"] = group["raw_low"].astype(float).rolling(
            10,
            min_periods=1,
        ).min()
        group["recent_swing_low"] = _recent_confirmed_swing_lows(
            group["raw_low"],
            pivot_k=pivot_k,
        ).fillna(group["rolling_low_10"])
        grouped = group
        for source, target in (
            ("raw_open", "next_open"),
            ("raw_high", "next_high"),
            ("raw_low", "next_low"),
            ("raw_close", "next_close"),
            ("trade_date", "execution_date"),
            ("is_suspended", "next_is_suspended"),
        ):
            grouped[target] = grouped[source].shift(-1)
        parts.append(grouped)
    result = pd.concat(parts, ignore_index=True)
    return result.loc[result["symbol"].isin(symbols)].sort_values(
        ["symbol", "trade_date"]
    ).reset_index(drop=True)


def _build_exit_paths(
    features: pd.DataFrame,
    symbols: list[str],
    entry_dates: list[date],
    holding_days_values: tuple[int, ...],
    *,
    lookahead_days: int,
) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    grouped = {
        symbol: group.sort_values("trade_date").reset_index(drop=True)
        for symbol, group in features.groupby("symbol", sort=False)
        if symbol in symbols
    }
    required_columns = [
        "raw_close",
        "raw_high",
        "raw_low",
        "atr_14_pct",
        "ma_10",
        "ma_20",
        "recent_swing_low",
        "next_open",
        "next_high",
        "next_low",
        "next_close",
        "execution_date",
    ]
    for entry_date in entry_dates:
        for symbol in symbols:
            symbol_frame = grouped.get(symbol)
            if symbol_frame is None:
                continue
            matches = symbol_frame.index[
                symbol_frame["trade_date"].dt.date == entry_date
            ].tolist()
            if not matches:
                continue
            start_index = matches[0]
            for holding_days in holding_days_values:
                row_count = int(holding_days) + int(lookahead_days)
                rows = symbol_frame.iloc[start_index : start_index + row_count].copy()
                if len(rows) != row_count:
                    continue
                if rows.iloc[:holding_days][required_columns].isna().any().any():
                    continue
                paths.append(
                    {
                        "entry_date": entry_date,
                        "symbol": symbol,
                        "holding_days": int(holding_days),
                        "entry_price": float(rows.iloc[0]["raw_close"]),
                        "rows": rows.reset_index(drop=True),
                    }
                )
    return paths


def _path_records(paths: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path_id": path_id,
                "entry_date": path["entry_date"],
                "symbol": path["symbol"],
                "holding_days": int(path["holding_days"]),
            }
            for path_id, path in enumerate(paths)
        ]
    )


def _atr(row: pd.Series, close: float) -> float:
    value = float(row.get("atr_14_pct", 0.0))
    if not np.isfinite(value) or value <= 0:
        return max(close * 0.02, 0.01)
    return max(close * value, 0.01)


def _hard_stop(cost: float, config: AppConfig) -> float:
    return float(cost) * (1.0 - float(config.positions.default_max_loss_pct))


def _profit_floor(cost: float, held_peak_high: float, config: AppConfig) -> float:
    peak_gain = held_peak_high / cost - 1.0
    settings = config.exit_machine
    if peak_gain < settings.profit_tier_1_gain:
        return -np.inf
    if peak_gain < settings.profit_tier_2_gain:
        return cost * (1.0 + settings.profit_tier_1_floor_pct)
    if peak_gain < settings.profit_tier_3_gain:
        return cost + settings.profit_tier_2_capture * (held_peak_high - cost)
    return cost + settings.profit_tier_3_capture * (held_peak_high - cost)


def _baseline_stop_for_row(
    *,
    cost: float,
    row: pd.Series,
    config: AppConfig,
) -> float:
    close = float(row["raw_close"])
    fixed_soft = close * (1.0 - float(config.exit_machine.fixed_baseline_gap))
    return max(_hard_stop(cost, config), fixed_soft)


def _trend_stop_for_row(
    *,
    held_peak_high: float,
    row: pd.Series,
    config: AppConfig,
) -> float:
    close = float(row["raw_close"])
    atr = _atr(row, close)
    swing_low = float(row["recent_swing_low"])
    chandelier_stop = held_peak_high - float(config.exit_machine.chandelier_mult) * atr
    swing_stop = swing_low - float(config.exit_machine.swing_trend_buffer) * atr
    return max(swing_stop, chandelier_stop)


def _round_stop(value: float, close: float, config: AppConfig) -> float:
    bounded = min(float(value), float(close))
    return round_to_tick(
        bounded,
        float(config.recommendation.price_tick),
        "down",
    )


def _post_exit_high_close(
    rows: pd.DataFrame,
    *,
    exit_day: int,
    days: int,
) -> float | None:
    start = int(exit_day)
    end = start + int(days)
    if len(rows) < end:
        return None
    values = pd.to_numeric(rows.iloc[start:end]["next_close"], errors="coerce")
    values = values.dropna()
    if len(values) != int(days):
        return None
    return float(values.max())


def _is_next_suspended(row: pd.Series) -> bool:
    value = row.get("next_is_suspended", False)
    if pd.isna(value):
        return False
    return bool(value)


def _base_result_fields(
    *,
    path_id: int,
    path: dict[str, Any],
    strategy: str,
    exit_day: int,
    exit_date: date | None,
    exit_price: float,
    exit_reason: str,
    exit_outcome: str,
    stop_update_count: int,
    unfilled_count: int,
    strategy_equity: list[float],
    strategy_adverse: list[float],
    held_peak_high: float,
    full_path_peak_gain: float,
    full_path_return: float,
    config: AppConfig,
) -> dict[str, Any]:
    entry_price = float(path["entry_price"])
    rows = path["rows"]
    holding_days = int(path["holding_days"])
    signal_rows = rows.iloc[:holding_days]
    hold_equity = [1.0] + (signal_rows["next_close"].astype(float) / entry_price).tolist()
    hold_adverse = [0.0] + (signal_rows["next_low"].astype(float) / entry_price - 1.0).tolist()
    strategy_return = float(exit_price / entry_price - 1.0)
    stopped = exit_reason != "hold_to_horizon"
    result = {
        "path_id": int(path_id),
        "entry_date": path["entry_date"],
        "symbol": path["symbol"],
        "holding_days": holding_days,
        "strategy": strategy,
        "entry_price": entry_price,
        "exit_day": int(exit_day),
        "exit_date": exit_date,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "exit_outcome": exit_outcome,
        "stopped": int(stopped),
        "stop_limit_unfilled_count": int(unfilled_count),
        "stop_update_count": int(stop_update_count),
        "strategy_return": strategy_return,
        "hold_return": full_path_return,
        "excess_return": strategy_return - full_path_return,
        "strategy_max_drawdown": _maximum_drawdown(strategy_equity),
        "hold_max_drawdown": _maximum_drawdown(hold_equity),
        "strategy_max_loss": float(min(strategy_adverse)),
        "hold_max_loss": float(min(hold_adverse)),
        "held_peak_gain": float(max(held_peak_high, exit_price) / entry_price - 1.0),
        "full_path_peak_gain": float(full_path_peak_gain),
        "full_path_return": float(full_path_return),
        "loss_avoided": int(stopped and strategy_return > full_path_return),
        "whipsaw": int(stopped and full_path_return - strategy_return >= 0.02),
    }
    for days in config.exit_machine.sell_fly_days:
        result[f"post_exit_high_close_{int(days)}d"] = _post_exit_high_close(
            rows,
            exit_day=exit_day,
            days=int(days),
        )
    return result


def _simulate_exit_machine_path(
    *,
    path_id: int,
    path: dict[str, Any],
    config: AppConfig,
) -> dict[str, Any]:
    rows = path["rows"]
    holding_days = int(path["holding_days"])
    signal_rows = rows.iloc[:holding_days].reset_index(drop=True)
    entry_price = float(path["entry_price"])
    tick = float(config.recommendation.price_tick)
    entry_row = signal_rows.iloc[0]
    entry_atr = _atr(entry_row, entry_price)
    baseline_stop_at_entry = _round_stop(
        _baseline_stop_for_row(
            cost=entry_price,
            row=entry_row,
            config=config,
        ),
        entry_price,
        config,
    )
    baseline_stop_at_entry = min(baseline_stop_at_entry, entry_price - tick)
    risk_r = max(
        entry_price - baseline_stop_at_entry,
        float(config.exit_machine.r_floor_atr_mult) * entry_atr,
        tick,
    )
    progress_price = entry_price + float(config.exit_machine.progress_r) * risk_r
    full_path_peak_high = max(
        [entry_price] + signal_rows["next_high"].astype(float).tolist()
    )
    full_path_peak_gain = full_path_peak_high / entry_price - 1.0
    full_path_return = float(signal_rows.iloc[-1]["next_close"]) / entry_price - 1.0

    active_stop: float | None = None
    trend_enabled = False
    held_peak_high = entry_price
    stop_update_count = 0
    unfilled_count = 0
    strategy_equity = [1.0]
    strategy_adverse = [0.0]

    for index, row in signal_rows.iterrows():
        close = float(row["raw_close"])
        atr = _atr(row, close)
        day_number = index + 1
        hard_stop = _hard_stop(entry_price, config)
        if trend_enabled:
            phase_stop = _trend_stop_for_row(
                held_peak_high=held_peak_high,
                row=row,
                config=config,
            )
            floor = _profit_floor(entry_price, held_peak_high, config)
        else:
            phase_stop = _baseline_stop_for_row(
                cost=entry_price,
                row=row,
                config=config,
            )
            floor = -np.inf

        technical_stop = max(hard_stop, phase_stop, floor)
        proposed_stop = _round_stop(technical_stop, close, config)
        previous_stop = active_stop
        active_stop = proposed_stop if active_stop is None else max(active_stop, proposed_stop)
        active_stop = min(active_stop, close)
        if previous_stop is None or active_stop >= previous_stop + tick:
            stop_update_count += 1

        if (
            not trend_enabled
            and held_peak_high >= progress_price
            and active_stop >= entry_price
        ):
            trend_enabled = True
            trend_stop = _trend_stop_for_row(
                held_peak_high=held_peak_high,
                row=row,
                config=config,
            )
            floor = _profit_floor(entry_price, held_peak_high, config)
            trend_proposed = _round_stop(max(hard_stop, trend_stop, floor), close, config)
            previous_stop = active_stop
            active_stop = max(active_stop, trend_proposed)
            active_stop = min(active_stop, close)
            if active_stop >= previous_stop + tick:
                stop_update_count += 1

        exit_reason: str | None = None
        if trend_enabled:
            ma20 = float(row["ma_20"])
            swing_low = float(row["recent_swing_low"])
            if (
                close < ma20 - float(config.exit_machine.breakdown_buffer) * atr
                and close < swing_low
            ):
                exit_reason = "technical_exit"

        if exit_reason is not None:
            exit_price = float(row["next_open"])
            exit_date = pd.Timestamp(row["execution_date"]).date()
            strategy_value = exit_price / entry_price
            strategy_equity.append(strategy_value)
            strategy_adverse.append(strategy_value - 1.0)
            return _base_result_fields(
                path_id=path_id,
                path=path,
                strategy=STATE_MACHINE_STRATEGY,
                exit_day=day_number,
                exit_date=exit_date,
                exit_price=exit_price,
                exit_reason=exit_reason,
                exit_outcome="NEXT_OPEN_EXIT",
                stop_update_count=stop_update_count,
                unfilled_count=unfilled_count,
                strategy_equity=strategy_equity,
                strategy_adverse=strategy_adverse,
                held_peak_high=held_peak_high,
                full_path_peak_gain=full_path_peak_gain,
                full_path_return=full_path_return,
                config=config,
            )

        stop_limit = round_to_tick(
            active_stop * (1.0 - float(config.exit_machine.limit_slippage_pct)),
            tick,
            "down",
        )
        simulation = simulate_next_day(
            close_price=close,
            take_profit_price=None,
            stop_trigger_price=float(active_stop),
            stop_limit_price=stop_limit,
            next_open=float(row["next_open"]),
            next_high=float(row["next_high"]),
            next_low=float(row["next_low"]),
            next_close=float(row["next_close"]),
            next_is_suspended=_is_next_suspended(row),
            stop_order_type=config.exit_machine.stop_order_type,
        )
        event = str(simulation["outcome"])
        if event == "STOP_LIMIT_UNFILLED":
            unfilled_count += 1
        exited = event not in {
            "NO_TRIGGER_CLOSE",
            "SUSPENDED_NO_TRIGGER",
            "STOP_LIMIT_UNFILLED",
        }
        if exited:
            exit_price = float(simulation["exit_price"])
            exit_date = pd.Timestamp(row["execution_date"]).date()
            strategy_value = exit_price / entry_price
            strategy_equity.append(strategy_value)
            strategy_adverse.append(strategy_value - 1.0)
            return _base_result_fields(
                path_id=path_id,
                path=path,
                strategy=STATE_MACHINE_STRATEGY,
                exit_day=day_number,
                exit_date=exit_date,
                exit_price=exit_price,
                exit_reason="intraday_stop",
                exit_outcome=event,
                stop_update_count=stop_update_count,
                unfilled_count=unfilled_count,
                strategy_equity=strategy_equity,
                strategy_adverse=strategy_adverse,
                held_peak_high=held_peak_high,
                full_path_peak_gain=full_path_peak_gain,
                full_path_return=full_path_return,
                config=config,
            )

        strategy_value = float(row["next_close"]) / entry_price
        strategy_equity.append(strategy_value)
        strategy_adverse.append(float(row["next_low"]) / entry_price - 1.0)
        held_peak_high = max(held_peak_high, float(row["next_high"]))

    exit_price = float(signal_rows.iloc[-1]["next_close"])
    exit_date = pd.Timestamp(signal_rows.iloc[-1]["execution_date"]).date()
    return _base_result_fields(
        path_id=path_id,
        path=path,
        strategy=STATE_MACHINE_STRATEGY,
        exit_day=holding_days,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason="hold_to_horizon",
        exit_outcome="HOLD_TO_HORIZON",
        stop_update_count=stop_update_count,
        unfilled_count=unfilled_count,
        strategy_equity=strategy_equity,
        strategy_adverse=strategy_adverse,
        held_peak_high=max(held_peak_high, exit_price),
        full_path_peak_gain=full_path_peak_gain,
        full_path_return=full_path_return,
        config=config,
    )


def _simulate_fixed_baseline_path(
    *,
    path_id: int,
    path: dict[str, Any],
    config: AppConfig,
) -> dict[str, Any]:
    rows = path["rows"]
    holding_days = int(path["holding_days"])
    signal_rows = rows.iloc[:holding_days].reset_index(drop=True)
    entry_price = float(path["entry_price"])
    tick = float(config.recommendation.price_tick)
    full_path_peak_high = max(
        [entry_price] + signal_rows["next_high"].astype(float).tolist()
    )
    full_path_peak_gain = full_path_peak_high / entry_price - 1.0
    full_path_return = float(signal_rows.iloc[-1]["next_close"]) / entry_price - 1.0

    active_stop: float | None = None
    held_peak_high = entry_price
    stop_update_count = 0
    unfilled_count = 0
    strategy_equity = [1.0]
    strategy_adverse = [0.0]

    for index, row in signal_rows.iterrows():
        close = float(row["raw_close"])
        proposed = _baseline_stop_for_row(
            cost=entry_price,
            row=row,
            config=config,
        )
        proposed = _round_stop(proposed, close, config)
        previous_stop = active_stop
        active_stop = proposed if active_stop is None else max(active_stop, proposed)
        active_stop = min(active_stop, close)
        if previous_stop is None or active_stop >= previous_stop + tick:
            stop_update_count += 1
        stop_limit = round_to_tick(
            active_stop * (1.0 - float(config.exit_machine.limit_slippage_pct)),
            tick,
            "down",
        )
        simulation = simulate_next_day(
            close_price=close,
            take_profit_price=None,
            stop_trigger_price=float(active_stop),
            stop_limit_price=stop_limit,
            next_open=float(row["next_open"]),
            next_high=float(row["next_high"]),
            next_low=float(row["next_low"]),
            next_close=float(row["next_close"]),
            next_is_suspended=_is_next_suspended(row),
            stop_order_type=config.exit_machine.stop_order_type,
        )
        event = str(simulation["outcome"])
        if event == "STOP_LIMIT_UNFILLED":
            unfilled_count += 1
        exited = event not in {
            "NO_TRIGGER_CLOSE",
            "SUSPENDED_NO_TRIGGER",
            "STOP_LIMIT_UNFILLED",
        }
        if exited:
            exit_price = float(simulation["exit_price"])
            exit_date = pd.Timestamp(row["execution_date"]).date()
            strategy_value = exit_price / entry_price
            strategy_equity.append(strategy_value)
            strategy_adverse.append(strategy_value - 1.0)
            return _base_result_fields(
                path_id=path_id,
                path=path,
                strategy=FIXED_BASELINE_STRATEGY,
                exit_day=index + 1,
                exit_date=exit_date,
                exit_price=exit_price,
                exit_reason="intraday_stop",
                exit_outcome=event,
                stop_update_count=stop_update_count,
                unfilled_count=unfilled_count,
                strategy_equity=strategy_equity,
                strategy_adverse=strategy_adverse,
                held_peak_high=held_peak_high,
                full_path_peak_gain=full_path_peak_gain,
                full_path_return=full_path_return,
                config=config,
            )
        strategy_value = float(row["next_close"]) / entry_price
        strategy_equity.append(strategy_value)
        strategy_adverse.append(float(row["next_low"]) / entry_price - 1.0)
        held_peak_high = max(held_peak_high, float(row["next_high"]))

    exit_price = float(signal_rows.iloc[-1]["next_close"])
    exit_date = pd.Timestamp(signal_rows.iloc[-1]["execution_date"]).date()
    return _base_result_fields(
        path_id=path_id,
        path=path,
        strategy=FIXED_BASELINE_STRATEGY,
        exit_day=holding_days,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason="hold_to_horizon",
        exit_outcome="HOLD_TO_HORIZON",
        stop_update_count=stop_update_count,
        unfilled_count=unfilled_count,
        strategy_equity=strategy_equity,
        strategy_adverse=strategy_adverse,
        held_peak_high=max(held_peak_high, exit_price),
        full_path_peak_gain=full_path_peak_gain,
        full_path_return=full_path_return,
        config=config,
    )


def _simulate_paths(paths: list[dict[str, Any]], config: AppConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path_id, path in enumerate(paths):
        records.append(
            _simulate_exit_machine_path(path_id=path_id, path=path, config=config)
        )
        records.append(
            _simulate_fixed_baseline_path(path_id=path_id, path=path, config=config)
        )
    return pd.DataFrame(records)


def _rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def summarize_exit_results(frame: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("没有可汇总的退出模拟结果")
    stopped = frame["stopped"].astype(bool)
    result: dict[str, Any] = {
        "path_count": int(len(frame)),
        "stopped_rate": float(frame["stopped"].mean()),
        "stop_limit_unfilled_rate": float(
            (frame["stop_limit_unfilled_count"] > 0).mean()
        ),
        "average_strategy_return": float(frame["strategy_return"].mean()),
        "average_hold_return": float(frame["hold_return"].mean()),
        "average_excess_return": float(frame["excess_return"].mean()),
        "p05_strategy_return": float(frame["strategy_return"].quantile(0.05)),
        "median_strategy_return": float(frame["strategy_return"].median()),
        "average_strategy_max_drawdown": float(frame["strategy_max_drawdown"].mean()),
        "average_hold_max_drawdown": float(frame["hold_max_drawdown"].mean()),
        "average_strategy_max_loss": float(frame["strategy_max_loss"].mean()),
        "average_hold_max_loss": float(frame["hold_max_loss"].mean()),
        "loss_avoidance_rate": (
            float(frame.loc[stopped, "loss_avoided"].mean()) if stopped.any() else 0.0
        ),
        "whipsaw_rate": (
            float(frame.loc[stopped, "whipsaw"].mean()) if stopped.any() else 0.0
        ),
        "trend_retention_rate": 0.0,
        "profit_giveback_mean": 0.0,
        "bad_exit_by_init_rate": 0.0,
        "bad_average_exit_day": None,
        "average_holding_days": float(frame["exit_day"].mean()),
    }
    winners = frame["full_path_peak_gain"].astype(float) >= float(
        config.exit_machine.winner_peak_gain
    )
    if winners.any():
        captured = frame["strategy_return"].astype(float) >= (
            float(config.exit_machine.trend_capture_ratio)
            * frame["full_path_peak_gain"].astype(float)
        )
        result["trend_retention_rate"] = float(captured.loc[winners].mean())
    positive_peak = frame["held_peak_gain"].astype(float) > 0
    if positive_peak.any():
        giveback = (
            frame.loc[positive_peak, "held_peak_gain"].astype(float)
            - frame.loc[positive_peak, "strategy_return"].astype(float)
        ) / frame.loc[positive_peak, "held_peak_gain"].astype(float)
        result["profit_giveback_mean"] = float(giveback.mean())
    bad = frame["full_path_return"].astype(float) < 0
    if bad.any():
        result["bad_exit_by_init_rate"] = float(
            (frame.loc[bad, "exit_day"].astype(int) <= int(config.exit_machine.n_init)).mean()
        )
        result["bad_average_exit_day"] = float(frame.loc[bad, "exit_day"].mean())

    for days in config.exit_machine.sell_fly_days:
        close_column = f"post_exit_high_close_{int(days)}d"
        observable = frame[close_column].notna()
        exited_observable = observable & stopped
        for threshold in config.exit_machine.sell_fly_thresholds:
            label = _sell_fly_label(days=int(days), threshold=float(threshold))
            fly = (
                observable
                & stopped
                & (
                    frame[close_column].astype(float)
                    >= frame["exit_price"].astype(float) * (1.0 + float(threshold))
                )
            )
            result[f"sell_fly_all_{label}"] = (
                float(fly.sum() / observable.sum()) if observable.any() else 0.0
            )
            result[f"sell_fly_exited_{label}"] = (
                float(fly.sum() / exited_observable.sum())
                if exited_observable.any()
                else 0.0
            )
    return result


def _sell_fly_label(*, days: int, threshold: float) -> str:
    return f"{int(days)}d_{int(round(float(threshold) * 100))}pct"


def _summarize_by_strategy(frame: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    return {
        strategy: summarize_exit_results(group, config)
        for strategy, group in frame.groupby("strategy", sort=True)
    }


def _delta_metrics(
    machine: dict[str, Any],
    baseline: dict[str, Any],
    config: AppConfig,
) -> dict[str, float]:
    primary_label = _sell_fly_label(
        days=int(config.exit_machine.primary_sell_fly_days),
        threshold=float(config.exit_machine.primary_sell_fly_threshold),
    )
    metrics = [
        "average_strategy_return",
        "p05_strategy_return",
        "average_strategy_max_loss",
        "whipsaw_rate",
        f"sell_fly_all_{primary_label}",
        "trend_retention_rate",
        "profit_giveback_mean",
        "bad_exit_by_init_rate",
        "average_holding_days",
    ]
    deltas: dict[str, float] = {}
    for metric in metrics:
        left = machine.get(metric)
        right = baseline.get(metric)
        if left is None or right is None:
            continue
        deltas[metric] = float(left) - float(right)
    if machine.get("bad_average_exit_day") is not None and baseline.get(
        "bad_average_exit_day"
    ) is not None:
        deltas["bad_average_exit_day"] = float(machine["bad_average_exit_day"]) - float(
            baseline["bad_average_exit_day"]
        )
    return deltas


def _window_diagnostics(
    summaries: dict[str, Any],
    config: AppConfig,
) -> dict[str, Any]:
    machine = summaries[STATE_MACHINE_STRATEGY]
    baseline = summaries[FIXED_BASELINE_STRATEGY]
    deltas = _delta_metrics(machine, baseline, config)
    tolerance = float(config.exit_machine.hard_constraint_tolerance_pct)
    primary_label = _sell_fly_label(
        days=int(config.exit_machine.primary_sell_fly_days),
        threshold=float(config.exit_machine.primary_sell_fly_threshold),
    )
    main = {
        "sell_fly_rate_lower": deltas.get(f"sell_fly_all_{primary_label}", 0.0) < 0.0,
        "trend_retention_rate_higher": deltas.get("trend_retention_rate", 0.0) > 0.0,
        "profit_giveback_lower": deltas.get("profit_giveback_mean", 0.0) < 0.0,
        "bad_exit_by_init_rate_higher": deltas.get("bad_exit_by_init_rate", 0.0) > 0.0,
    }
    hard = {
        "average_max_loss_not_worse_0_5pp": deltas.get(
            "average_strategy_max_loss",
            0.0,
        )
        >= -tolerance,
        "p05_return_not_worse_0_5pp": deltas.get("p05_strategy_return", 0.0)
        >= -tolerance,
    }
    return {
        "pass": bool(all(hard.values()) and sum(main.values()) >= 3),
        "hard_constraints": hard,
        "main_edges": main,
        "main_edge_count": int(sum(main.values())),
        "deltas": deltas,
    }


def _entry_months(samples: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(samples["entry_date"]).dt.to_period("M").astype(str)


def _walk_forward_windows(samples: pd.DataFrame, config: AppConfig) -> list[dict[str, Any]]:
    months = sorted(_entry_months(samples).unique())
    train_months = int(config.exit_machine.walk_forward_train_months)
    validation_months = int(config.exit_machine.validation_months)
    step_months = int(config.exit_machine.walk_forward_step_months)
    if len(months) < train_months + validation_months:
        return []
    entry_months = _entry_months(samples)
    windows: list[dict[str, Any]] = []
    window_no = 1
    for validation_start in range(
        train_months,
        len(months) - validation_months + 1,
        step_months,
    ):
        train_values = months[validation_start - train_months : validation_start]
        validation_values = months[validation_start : validation_start + validation_months]
        windows.append(
            {
                "window_id": f"wf_{window_no:02d}",
                "train_months": train_values,
                "validation_months": validation_values,
                "validation_path_ids": set(
                    samples.loc[entry_months.isin(validation_values), "path_id"]
                    .astype(int)
                    .tolist()
                ),
            }
        )
        window_no += 1
    return windows


def _bootstrap_deltas(
    frame: pd.DataFrame,
    config: AppConfig,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0:
        return {"status": "skipped", "reason": "bootstrap disabled"}
    rng = np.random.default_rng(seed)
    path_ids = frame["path_id"].astype(int).drop_duplicates().to_numpy()
    if len(path_ids) < 2:
        return {"status": "skipped", "reason": "not enough paths"}
    by_path = {
        int(path_id): frame.index[frame["path_id"].astype(int) == int(path_id)].to_numpy()
        for path_id in path_ids
    }
    collected: dict[str, list[float]] = {}
    pass_count = 0
    for _ in range(iterations):
        sampled = rng.choice(path_ids, size=len(path_ids), replace=True)
        indices = np.concatenate([by_path[int(path_id)] for path_id in sampled])
        summaries = _summarize_by_strategy(frame.loc[indices], config)
        diagnostics = _window_diagnostics(summaries, config)
        pass_count += int(diagnostics["pass"])
        for metric, value in diagnostics["deltas"].items():
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


def _directional_pass(metric: str, mean: float, se: float, config: AppConfig) -> bool:
    primary_label = _sell_fly_label(
        days=int(config.exit_machine.primary_sell_fly_days),
        threshold=float(config.exit_machine.primary_sell_fly_threshold),
    )
    lower_is_better = {
        "whipsaw_rate",
        f"sell_fly_all_{primary_label}",
        "profit_giveback_mean",
        "bad_average_exit_day",
        "average_holding_days",
    }
    if metric in lower_is_better:
        return bool(mean < -se)
    return bool(mean > se)


def _aggregate_diagnostics(records: list[dict[str, Any]], config: AppConfig) -> dict[str, Any]:
    ok = [record for record in records if record.get("status") == "ok"]
    if not ok:
        return {"status": "skipped", "reason": "no successful windows"}
    delta_rows = [record["diagnostics"]["deltas"] for record in ok]
    deltas = pd.DataFrame(delta_rows).fillna(0.0)
    delta_summary: dict[str, Any] = {}
    for metric in deltas.columns:
        values = deltas[metric].astype(float)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        window_direction_rate = float(
            np.mean(
                [
                    _directional_pass(metric, float(value), 0.0, config)
                    for value in values
                ]
            )
        )
        delta_summary[metric] = {
            "mean": mean,
            "standard_error_across_windows": se,
            "window_direction_rate": window_direction_rate,
            "passes_directional_gate": bool(
                window_direction_rate >= 0.5
                and _directional_pass(metric, mean, se, config)
            ),
        }
    primary_label = _sell_fly_label(
        days=int(config.exit_machine.primary_sell_fly_days),
        threshold=float(config.exit_machine.primary_sell_fly_threshold),
    )
    tolerance = float(config.exit_machine.hard_constraint_tolerance_pct)
    hard = {
        "average_max_loss_not_worse_0_5pp": delta_summary[
            "average_strategy_max_loss"
        ]["mean"]
        >= -tolerance,
        "p05_return_not_worse_0_5pp": delta_summary["p05_strategy_return"]["mean"]
        >= -tolerance,
    }
    main_metric_names = [
        f"sell_fly_all_{primary_label}",
        "trend_retention_rate",
        "profit_giveback_mean",
        "bad_exit_by_init_rate",
    ]
    main_passes = {
        metric: bool(delta_summary.get(metric, {}).get("passes_directional_gate", False))
        for metric in main_metric_names
    }
    criteria = {
        **hard,
        "main_edges_at_least_three": sum(main_passes.values()) >= 3,
        "window_pass_rate_at_least_half": float(
            np.mean([record["diagnostics"]["pass"] for record in ok])
        )
        >= 0.5,
    }
    return {
        "status": "ok",
        "window_count": int(len(ok)),
        "pass_rate": float(np.mean([record["diagnostics"]["pass"] for record in ok])),
        "hard_constraints": hard,
        "main_edge_passes": main_passes,
        "criteria": criteria,
        "pass": bool(all(criteria.values())),
        "delta_summary": delta_summary,
    }


def _run_walk_forward(
    samples: pd.DataFrame,
    results: pd.DataFrame,
    config: AppConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    windows = _walk_forward_windows(samples, config)
    if not windows:
        return (
            {
                "status": "skipped",
                "reason": "not enough entry months for configured walk-forward",
            },
            pd.DataFrame(),
        )
    records: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for window in windows:
        validation_ids = window["validation_path_ids"]
        frame = results.loc[results["path_id"].astype(int).isin(validation_ids)].copy()
        if frame.empty:
            records.append(
                {
                    "window_id": window["window_id"],
                    "status": "failed",
                    "error": "empty validation frame",
                }
            )
            continue
        summaries = _summarize_by_strategy(frame, config)
        diagnostics = _window_diagnostics(summaries, config)
        record = {
            "window_id": window["window_id"],
            "status": "ok",
            "train_months": window["train_months"],
            "validation_months": window["validation_months"],
            "validation_path_count": int(len(validation_ids)),
            "summaries": summaries,
            "diagnostics": diagnostics,
            "bootstrap": _bootstrap_deltas(
                frame,
                config,
                iterations=int(config.exit_machine.bootstrap_samples),
                seed=int(config.training.random_seed),
            ),
        }
        records.append(record)
        frame["window_id"] = window["window_id"]
        frames.append(frame)
    aggregate = _aggregate_diagnostics(records, config)
    return (
        {
            "status": "ok" if frames else "failed",
            "records": records,
            "aggregate": aggregate,
            "pass": bool(aggregate.get("pass", False)),
        },
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
    )


def run_exit_machine_backtest(
    engine: Any,
    config: AppConfig,
    *,
    start_date: date,
    end_date: date,
    max_symbols: int | None = None,
    output_directory: str | Path = "output/exit-machine",
) -> dict[str, Any]:
    if start_date > end_date:
        raise ValueError("状态机退出回测开始日期不能晚于结束日期")
    max_symbols = max_symbols or config.exit_machine.max_symbols
    entry_dates = _entry_dates(engine, start_date, end_date)
    if not entry_dates:
        raise RuntimeError("状态机退出回测范围没有交易日")
    max_holding = max(config.exit_machine.holding_days)
    lookahead = int(config.exit_machine.lookahead_extend_days)
    complete_entry_dates: list[date] = []
    skipped_incomplete_entry_dates: list[date] = []
    for entry_date in entry_dates:
        try:
            _path_end_date(engine, entry_date, max_holding + lookahead)
            complete_entry_dates.append(entry_date)
        except RuntimeError:
            skipped_incomplete_entry_dates.append(entry_date)
    if not complete_entry_dates:
        raise RuntimeError(
            "状态机退出回测没有满足 holding_days + lookahead 的完整建仓月份"
        )
    path_end_date = _path_end_date(
        engine,
        complete_entry_dates[-1],
        max_holding + lookahead,
    )
    entry_dates = complete_entry_dates
    symbols = _select_evaluation_symbols(engine, config, start_date, max_symbols)
    if not symbols:
        raise RuntimeError("没有可用于状态机退出回测的股票")

    features = _prepare_exit_features(
        engine,
        config,
        entry_dates[0],
        path_end_date,
        symbols,
    )
    signal_features = features.loc[
        (features["trade_date"].dt.date >= entry_dates[0])
        & (features["trade_date"].dt.date < path_end_date)
    ].copy()
    paths = _build_exit_paths(
        signal_features,
        symbols,
        entry_dates,
        config.exit_machine.holding_days,
        lookahead_days=lookahead,
    )
    if not paths:
        raise RuntimeError("没有构建出完整的状态机退出回测路径")
    print(
        f"状态机退出回测：{len(symbols)} 支股票，{len(entry_dates)} 个建仓月，"
        f"{len(paths)} 条路径",
        flush=True,
    )

    results = _simulate_paths(paths, config)
    samples = _path_records(paths)
    summaries = _summarize_by_strategy(results, config)
    diagnostics = _window_diagnostics(summaries, config)
    walk_forward, walk_forward_paths = _run_walk_forward(samples, results, config)

    run_id = datetime.now(timezone.utc).strftime("em_%Y%m%dT%H%M%SZ")
    output_root = Path(output_directory).resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    paths_path = output_root / "paths.csv"
    results_path = output_root / "exit_paths.csv"
    walk_forward_path = output_root / "walk_forward_summary.json"
    walk_forward_paths_path = output_root / "walk_forward_paths.csv"
    summary_path = output_root / "summary.json"
    samples.to_csv(paths_path, index=False, encoding="utf-8-sig")
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    if not walk_forward_paths.empty:
        walk_forward_paths.to_csv(walk_forward_paths_path, index=False, encoding="utf-8-sig")
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
        "skipped_incomplete_entry_dates": [
            value.isoformat() for value in skipped_incomplete_entry_dates
        ],
        "path_count": int(len(paths)),
        "sample_count": int(len(samples)),
        "summaries": summaries,
        "diagnostics": diagnostics,
        "walk_forward": walk_forward,
        "promotion_action": "report_only",
        "production_changed": False,
        "exit_machine_config": asdict(config.exit_machine),
        "output_directory": str(output_root),
        "paths_path": str(paths_path),
        "exit_paths_path": str(results_path),
        "walk_forward_summary_path": str(walk_forward_path),
        "walk_forward_paths_path": (
            str(walk_forward_paths_path) if not walk_forward_paths.empty else None
        ),
        "method": (
            "状态机退出 v2：盈利门前逐字复用固定 2% + 硬线 + 自维护棘轮；"
            "达到 +1R 且止损抬到成本上方后，才启用宽 trend stop、次日开盘确认退出"
            "和分段利润底。所有结果 report-only，不读取旧 current_stop，不改生产配置。"
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary
