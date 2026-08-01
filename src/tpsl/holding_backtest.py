from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from .backtest import (
    _previous_trading_date,
    _train_backtest_fold,
    simulate_next_day,
)
from .config import AppConfig
from .db import load_bars, load_positions
from .features import FEATURE_COLUMNS, build_feature_frame
from .model import ModelBundle
from .risk import make_risk_decision, round_to_tick
from .similarity import SimilarityIndex


def _maximum_drawdown(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(array)
    return float(np.min(array / peaks - 1.0))


def ratchet_stop(previous_stop: float | None, proposed_stop: float) -> float:
    return (
        proposed_stop
        if previous_stop is None
        else max(previous_stop, proposed_stop)
    )


def _fold_root(config: AppConfig) -> Path:
    fingerprint = hashlib.sha1(
        json.dumps(
            asdict(config.training),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return (
        config.artifacts_directory
        / "backtests"
        / "folds"
        / fingerprint
    )


def _prepare_position_features(
    engine: Any,
    config: AppConfig,
    earliest_entry: date,
    end_date: date,
    symbols: list[str],
) -> pd.DataFrame:
    feature_start = earliest_entry - timedelta(days=260)
    bars = load_bars(engine, config, feature_start, end_date)
    if bars.empty:
        raise RuntimeError("真实持仓回测范围没有行情数据")
    print(f"持仓回测特征行情读取完成：{len(bars):,} 行", flush=True)
    features = build_feature_frame(
        bars,
        workers=config.performance.workers,
        min_symbol_rows=25,
    )
    grouped = features.groupby("symbol", sort=False)
    for source, target in (
        ("raw_open", "next_open"),
        ("raw_high", "next_high"),
        ("raw_low", "next_low"),
        ("raw_close", "next_close"),
        ("trade_date", "execution_date"),
        ("is_suspended", "next_is_suspended"),
    ):
        features[target] = grouped[source].shift(-1)
    return features.loc[features["symbol"].isin(symbols)].copy()


def _predict_stops(
    engine: Any,
    config: AppConfig,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    usable = frame.dropna(
        subset=FEATURE_COLUMNS + [
            "atr_14_pct",
            "raw_close",
            "next_open",
            "next_high",
            "next_low",
            "next_close",
            "execution_date",
        ]
    ).copy()
    if usable.empty:
        return usable
    usable["month"] = usable["trade_date"].dt.to_period("M").astype(str)
    predicted_parts: list[pd.DataFrame] = []
    root = _fold_root(config)

    for month, month_frame in usable.groupby("month", sort=True):
        first_signal_date = month_frame["trade_date"].min().date()
        train_end_date = _previous_trading_date(engine, first_signal_date)
        fold_directory = root / train_end_date.isoformat()
        print(
            f"持仓回测月份 {month}：模型训练截止 {train_end_date}",
            flush=True,
        )
        _train_backtest_fold(
            engine,
            config,
            train_end_date,
            fold_directory,
            reuse=True,
        )
        bundle = ModelBundle(fold_directory)
        similarity = SimilarityIndex(fold_directory)
        month_frame = month_frame.copy()
        month_frame["predicted_high_return"] = bundle.predict(
            month_frame,
            "high",
            config.recommendation.take_profit_quantile,
        )
        month_frame["predicted_low_return"] = bundle.predict(
            month_frame,
            "low",
            config.recommendation.stop_loss_quantile,
        )
        neighbors = similarity.query_batch(
            month_frame,
            config.recommendation.similarity_top_k,
            sample_limit=config.backtest.similarity_query_sample_limit,
            batch_size=config.backtest.similarity_batch_size,
        )
        month_frame = month_frame.join(neighbors)
        month_frame["fold_train_end_date"] = train_end_date
        month_frame["model_version"] = bundle.version
        predicted_parts.append(month_frame)
    return pd.concat(predicted_parts, ignore_index=True)


def _last_close(
    engine: Any,
    symbol: str,
    end_date: date,
) -> tuple[date, float]:
    query = text(
        """
        SELECT trade_date, close
        FROM stock_daily_bars
        WHERE symbol = :symbol AND trade_date <= :end_date
        ORDER BY trade_date DESC
        LIMIT 1
        """
    )
    with engine.connect() as connection:
        row = connection.execute(
            query,
            {"symbol": symbol, "end_date": end_date},
        ).one_or_none()
    if row is None:
        raise RuntimeError(f"{symbol} 在 {end_date} 前没有行情")
    return row[0], float(row[1])


def _simulate_position(
    *,
    run_id: str,
    position: pd.Series,
    rows: pd.DataFrame,
    end_date: date,
    end_trade_date: date,
    end_close: float,
    config: AppConfig,
    stop_order_type: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    symbol = str(position["symbol"])
    avg_cost = float(position["avg_cost"])
    quantity = float(position["quantity"])
    max_loss_pct = position.get("max_loss_pct")
    if pd.isna(max_loss_pct) or not 0 < float(max_loss_pct) < 1:
        max_loss_pct = config.positions.default_max_loss_pct

    active_stop: float | None = None
    stop_update_count = 0
    unfilled_count = 0
    exit_date: date | None = None
    exit_price: float | None = None
    exit_outcome = "OPEN_AT_END"
    strategy_equity = [1.0]
    strategy_adverse = [0.0]
    daily_records: list[dict[str, Any]] = []
    full_rows = rows.sort_values("trade_date")
    hold_equity = [1.0] + (
        full_rows["next_close"].astype(float) / avg_cost
    ).tolist()
    hold_adverse = [0.0] + (
        full_rows["next_low"].astype(float) / avg_cost - 1.0
    ).tolist()

    for _, row in full_rows.iterrows():
        close = float(row["raw_close"])
        atr = close * float(row["atr_14_pct"])
        decision = make_risk_decision(
            close=close,
            avg_cost=avg_cost,
            max_loss_pct=float(max_loss_pct),
            current_stop=active_stop,
            predicted_high_return=float(row["predicted_high_return"]),
            predicted_low_return=float(row["predicted_low_return"]),
            similar_high_return=float(row["median_high_return"]),
            similar_low_return=float(row["low_return_q20"]),
            atr_14=atr,
            similarity_distance=float(row["mean_distance"]),
            sample_count=int(row["sample_count"]),
            config=config.recommendation,
        )
        proposed_stop = decision.stop_trigger_price
        previous_stop = active_stop
        active_stop = ratchet_stop(active_stop, proposed_stop)
        if (
            previous_stop is None
            or active_stop >= previous_stop + config.recommendation.price_tick
        ):
            stop_update_count += 1
        active_limit = round_to_tick(
            active_stop
            * (1 - config.recommendation.stop_limit_slippage_pct),
            config.recommendation.price_tick,
            "down",
        )

        simulation = simulate_next_day(
            close_price=close,
            take_profit_price=None,
            stop_trigger_price=active_stop,
            stop_limit_price=active_limit,
            next_open=float(row["next_open"]),
            next_high=float(row["next_high"]),
            next_low=float(row["next_low"]),
            next_close=float(row["next_close"]),
            next_is_suspended=bool(row.get("next_is_suspended", False)),
            stop_order_type=stop_order_type,
        )
        event = str(simulation["outcome"])
        execution_date = pd.Timestamp(row["execution_date"]).date()
        hold_value = float(row["next_close"]) / avg_cost

        exited = event not in {
            "NO_TRIGGER_CLOSE",
            "SUSPENDED_NO_TRIGGER",
            "STOP_LIMIT_UNFILLED",
        }
        if event == "STOP_LIMIT_UNFILLED":
            unfilled_count += 1
        if exited:
            exit_date = execution_date
            exit_price = float(simulation["exit_price"])
            exit_outcome = event
            strategy_value = exit_price / avg_cost
            strategy_adverse.append(strategy_value - 1)
        else:
            strategy_value = float(row["next_close"]) / avg_cost
            strategy_adverse.append(float(row["next_low"]) / avg_cost - 1)
        strategy_equity.append(strategy_value)

        daily_records.append(
            {
                "run_id": run_id,
                "symbol": symbol,
                "signal_date": row["trade_date"].date(),
                "execution_date": execution_date,
                "close_price": close,
                "proposed_stop_price": proposed_stop,
                "active_stop_price": active_stop,
                "stop_limit_price": active_limit,
                "next_open": float(row["next_open"]),
                "next_high": float(row["next_high"]),
                "next_low": float(row["next_low"]),
                "next_close": float(row["next_close"]),
                "event": event,
                "strategy_equity": strategy_value,
                "hold_equity": hold_value,
                "confidence": decision.confidence,
                "model_version": str(row["model_version"]),
            }
        )
        if exited:
            break

    hold_return = end_close / avg_cost - 1
    if exit_price is not None:
        strategy_return = exit_price / avg_cost - 1
        # 退出后保持现金，补齐到回测截止日用于回撤比较。
        strategy_equity.append(strategy_equity[-1])
    else:
        strategy_return = hold_return
        if not rows.empty:
            final_known_date = pd.Timestamp(
                rows["execution_date"].max()
            ).date()
        else:
            final_known_date = position["entry_date"]
        if final_known_date < end_trade_date:
            strategy_equity.append(end_close / avg_cost)
            strategy_adverse.append(end_close / avg_cost - 1)
        exit_outcome = "OPEN_AT_END"

    # 持有基准始终评估到统一截止日。
    if not math.isclose(
        hold_equity[-1],
        end_close / avg_cost,
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        hold_equity.append(end_close / avg_cost)
        hold_adverse.append(end_close / avg_cost - 1)

    summary = {
        "run_id": run_id,
        "symbol": symbol,
        "entry_date": position["entry_date"],
        "avg_cost": avg_cost,
        "quantity": quantity,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "exit_outcome": exit_outcome,
        "final_stop_price": active_stop,
        "strategy_return": strategy_return,
        "hold_return": hold_return,
        "excess_return": strategy_return - hold_return,
        "strategy_max_drawdown": _maximum_drawdown(strategy_equity),
        "hold_max_drawdown": _maximum_drawdown(hold_equity),
        "strategy_max_loss_from_cost": float(min(strategy_adverse)),
        "hold_max_loss_from_cost": float(min(hold_adverse)),
        "stop_update_count": stop_update_count,
        "stop_limit_unfilled_count": unfilled_count,
        "trading_days": len(daily_records),
    }
    return summary, daily_records


def run_holding_backtest(
    engine: Any,
    config: AppConfig,
    *,
    end_date: date,
    stop_order_type: str | None = None,
    output_directory: str | Path = "output/holding-backtests",
    write_database: bool = True,
) -> dict[str, Any]:
    stop_order_type = stop_order_type or config.backtest.stop_order_type
    positions = load_positions(engine, config)
    positions = positions.dropna(subset=["entry_date"]).copy()
    if positions.empty:
        raise RuntimeError("当前有效持仓没有填写 entry_date")
    positions["entry_date"] = pd.to_datetime(
        positions["entry_date"]
    ).dt.date
    positions = positions.loc[positions["entry_date"] <= end_date].copy()
    if positions.empty:
        raise RuntimeError("没有建仓日期早于回测截止日的持仓")

    run_id = datetime.now(timezone.utc).strftime("hbt_%Y%m%dT%H%M%SZ")
    output_root = Path(output_directory).resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    earliest_entry = min(positions["entry_date"])
    symbols = sorted(positions["symbol"].astype(str).unique().tolist())
    features = _prepare_position_features(
        engine,
        config,
        earliest_entry,
        end_date,
        symbols,
    )
    signal_rows = features.loc[
        (features["trade_date"].dt.date >= earliest_entry)
        & (features["trade_date"].dt.date <= end_date)
    ].copy()
    predicted = _predict_stops(engine, config, signal_rows)

    position_summaries: list[dict[str, Any]] = []
    daily_records: list[dict[str, Any]] = []
    for _, position in positions.iterrows():
        symbol = str(position["symbol"])
        entry_date = position["entry_date"]
        rows = predicted.loc[
            (predicted["symbol"] == symbol)
            & (predicted["trade_date"].dt.date >= entry_date)
            & (predicted["trade_date"].dt.date < end_date)
        ].copy()
        end_trade_date, end_close = _last_close(engine, symbol, end_date)
        summary, daily = _simulate_position(
            run_id=run_id,
            position=position,
            rows=rows,
            end_date=end_date,
            end_trade_date=end_trade_date,
            end_close=end_close,
            config=config,
            stop_order_type=stop_order_type,
        )
        position_summaries.append(summary)
        daily_records.extend(daily)

    position_frame = pd.DataFrame(position_summaries)
    daily_frame = pd.DataFrame(daily_records)
    aggregate = {
        "position_count": int(len(position_frame)),
        "stopped_positions": int(position_frame["exit_date"].notna().sum()),
        "average_strategy_return": float(
            position_frame["strategy_return"].mean()
        ),
        "average_hold_return": float(position_frame["hold_return"].mean()),
        "average_excess_return": float(
            position_frame["excess_return"].mean()
        ),
        "average_strategy_max_drawdown": float(
            position_frame["strategy_max_drawdown"].mean()
        ),
        "average_hold_max_drawdown": float(
            position_frame["hold_max_drawdown"].mean()
        ),
        "average_strategy_max_loss_from_cost": float(
            position_frame["strategy_max_loss_from_cost"].mean()
        ),
        "average_hold_max_loss_from_cost": float(
            position_frame["hold_max_loss_from_cost"].mean()
        ),
    }
    result = {
        "run_id": run_id,
        "start_date": earliest_entry.isoformat(),
        "end_date": end_date.isoformat(),
        "stop_order_type": stop_order_type,
        "metrics": aggregate,
        "positions": position_summaries,
        "limitations": [
            "使用日线模拟止损成交",
            "止损只升不降",
            "止损限价跳空未成交时继续持仓",
            "使用当前持仓表中的建仓日期、成本和数量",
        ],
    }

    position_frame.to_csv(
        output_root / "positions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily_frame.to_csv(
        output_root / "daily.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if write_database:
        from .db import save_holding_backtest_result

        save_holding_backtest_result(
            engine,
            result=result,
            config_json={
                "training": asdict(config.training),
                "recommendation": asdict(config.recommendation),
                "backtest": asdict(config.backtest),
            },
            positions=position_frame,
            daily=daily_frame,
        )
    result["output_directory"] = str(output_root)
    return result
