from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from .config import AppConfig
from .db import load_bars, load_positions, select_training_symbols
from .features import FEATURE_COLUMNS, build_feature_frame
from .model import ModelBundle, train_models
from .risk import make_risk_decision, round_to_tick
from .similarity import SimilarityIndex, build_similarity_index


STRATEGY_HYBRID = "hybrid"
STRATEGY_FIXED = "fixed_3_2"
STRATEGY_ATR = "atr"
STRATEGY_HOLD = "hold_close"


def simulate_next_day(
    *,
    close_price: float,
    take_profit_price: float | None,
    stop_trigger_price: float,
    stop_limit_price: float,
    next_open: float,
    next_high: float,
    next_low: float,
    next_close: float,
    next_is_suspended: bool,
    stop_order_type: str,
) -> dict[str, Any]:
    if stop_order_type not in {"market", "limit"}:
        raise ValueError("stop_order_type 必须是 market 或 limit")

    hit_take_profit = (
        take_profit_price is not None
        and next_high >= take_profit_price
    )
    hit_stop = next_low <= stop_trigger_price
    dual_hit = hit_take_profit and hit_stop
    gap_stop = next_open <= stop_trigger_price
    stop_limit_unfilled = False

    if next_is_suspended:
        exit_price = next_close
        outcome = "SUSPENDED_NO_TRIGGER"
    elif gap_stop:
        if stop_order_type == "market":
            exit_price = next_open
            outcome = "GAP_STOP_MARKET"
        elif next_open >= stop_limit_price:
            exit_price = next_open
            outcome = "GAP_STOP_LIMIT_FILLED"
        elif next_high >= stop_limit_price:
            exit_price = stop_limit_price
            outcome = "GAP_STOP_LIMIT_RECOVERED"
        else:
            exit_price = next_close
            outcome = "STOP_LIMIT_UNFILLED"
            stop_limit_unfilled = True
    elif hit_stop:
        # 日线无法判断同日先后，保守假设止损先触发。
        exit_price = (
            stop_trigger_price
            if stop_order_type == "market"
            else stop_limit_price
        )
        outcome = "DUAL_HIT_STOP_FIRST" if dual_hit else "STOP_LOSS"
    elif (
        take_profit_price is not None
        and next_open >= take_profit_price
    ):
        exit_price = next_open
        outcome = "GAP_TAKE_PROFIT"
    elif hit_take_profit:
        exit_price = take_profit_price
        outcome = "TAKE_PROFIT"
    else:
        exit_price = next_close
        outcome = "NO_TRIGGER_CLOSE"

    return {
        "exit_price": float(exit_price),
        "outcome": outcome,
        "return_pct": float(exit_price / close_price - 1),
        "stop_limit_unfilled": int(stop_limit_unfilled),
        "dual_hit": int(dual_hit),
        "gap_stop": int(gap_stop),
    }


def calculate_backtest_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"trade_count": 0}

    returns = trades["return_pct"].astype(float)
    gains = returns.loc[returns > 0].sum()
    losses = -returns.loc[returns < 0].sum()
    daily = (
        trades.groupby("signal_date", sort=True)["return_pct"]
        .mean()
        .astype(float)
    )
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0

    max_consecutive_losses = 0
    current_losses = 0
    for value in daily:
        if value < 0:
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_losses = 0

    outcomes = trades["outcome"].astype(str)
    take_profit_mask = outcomes.str.contains("TAKE_PROFIT")
    stop_mask = outcomes.str.contains("STOP")
    no_trigger_mask = outcomes.isin(
        ["NO_TRIGGER_CLOSE", "SUSPENDED_NO_TRIGGER", "HOLD_CLOSE"]
    )
    return {
        "trade_count": int(len(trades)),
        "signal_days": int(daily.index.nunique()),
        "average_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "win_rate": float((returns > 0).mean()),
        "loss_rate": float((returns < 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "take_profit_rate": float(take_profit_mask.mean()),
        "stop_rate": float(stop_mask.mean()),
        "no_trigger_rate": float(no_trigger_mask.mean()),
        "dual_hit_rate": float(trades["dual_hit"].mean()),
        "gap_stop_rate": float(trades["gap_stop"].mean()),
        "stop_limit_unfilled_rate": float(
            trades["stop_limit_unfilled"].mean()
        ),
        "equal_weight_total_return": float(equity.iloc[-1] - 1),
        "maximum_drawdown": float(drawdown.min()),
        "daily_sharpe": (
            float(daily.mean() / daily_std * math.sqrt(252))
            if daily_std > 0
            else None
        ),
        "max_consecutive_losing_days": int(max_consecutive_losses),
    }


def _previous_trading_date(
    engine: Any,
    before_date: date,
) -> date:
    query = text(
        """
        SELECT MAX(trade_date)
        FROM stock_daily_bars
        WHERE trade_date < :before_date
        """
    )
    with engine.connect() as connection:
        value = connection.execute(
            query,
            {"before_date": before_date},
        ).scalar_one()
    if value is None:
        raise RuntimeError(f"{before_date} 之前没有可用交易日")
    return value


def _select_evaluation_symbols(
    engine: Any,
    config: AppConfig,
    start_date: date,
    max_symbols: int,
) -> list[str]:
    history_start = start_date - timedelta(days=365)
    query = text(
        """
        SELECT symbol, COUNT(*) AS row_count
        FROM stock_daily_bars
        WHERE trade_date BETWEEN :history_start AND :start_date
        GROUP BY symbol
        HAVING COUNT(*) >= :min_rows
        ORDER BY symbol
        """
    )
    eligible = pd.read_sql(
        query,
        engine,
        params={
            "history_start": history_start,
            "start_date": start_date,
            "min_rows": config.training.min_symbol_rows,
        },
    )
    if eligible.empty:
        raise RuntimeError("没有满足历史长度要求的回测股票")

    positions = load_positions(engine, config)
    forced = (
        set(positions["symbol"].astype(str))
        if not positions.empty
        else set()
    )
    eligible_symbols = eligible["symbol"].astype(str)
    forced = forced.intersection(set(eligible_symbols))
    if len(eligible) <= max_symbols:
        return sorted(eligible_symbols.tolist())

    slots = max(0, max_symbols - len(forced))
    candidates = eligible.loc[~eligible_symbols.isin(forced)]
    sampled = candidates.sample(
        n=min(slots, len(candidates)),
        random_state=config.training.random_seed + 1000,
    )["symbol"].astype(str)
    return sorted(forced.union(set(sampled)))


def _prepare_evaluation_frame(
    engine: Any,
    config: AppConfig,
    start_date: date,
    end_date: date,
    symbols: list[str],
) -> pd.DataFrame:
    feature_start = start_date - timedelta(days=260)
    next_date_query = text(
        "SELECT MIN(trade_date) FROM stock_daily_bars WHERE trade_date > :end_date"
    )
    with engine.connect() as connection:
        next_date = connection.execute(
            next_date_query,
            {"end_date": end_date},
        ).scalar_one()
    load_end_date = next_date or end_date
    bars = load_bars(engine, config, feature_start, load_end_date)
    if bars.empty:
        raise RuntimeError("回测日期范围没有行情数据")
    print(f"回测特征行情读取完成：{len(bars):,} 行", flush=True)
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

    signal_mask = (
        features["trade_date"].dt.date >= start_date
    ) & (features["trade_date"].dt.date <= end_date)
    selected = features.loc[
        signal_mask & features["symbol"].isin(symbols)
    ].copy()
    required = FEATURE_COLUMNS + [
        "atr_14_pct",
        "raw_close",
        "next_open",
        "next_high",
        "next_low",
        "next_close",
        "execution_date",
    ]
    selected = selected.dropna(subset=required)
    if selected.empty:
        raise RuntimeError("回测范围没有完整的信号日和次日行情")
    selected["month"] = selected["trade_date"].dt.to_period("M").astype(str)
    return selected


def _train_backtest_fold(
    engine: Any,
    config: AppConfig,
    train_end_date: date,
    fold_directory: Path,
    reuse: bool,
) -> dict[str, Any]:
    metadata_path = fold_directory / "metadata.json"
    required_files = [
        metadata_path,
        fold_directory / "similarity_index.npz",
        fold_directory / "high_q70.json",
        fold_directory / "low_q20.json",
    ]
    if reuse and all(path.exists() for path in required_files):
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    fold_config = replace(config, artifacts_directory=fold_directory)
    training_start = train_end_date - timedelta(
        days=config.training.lookback_days * 2
    )
    training_symbols = select_training_symbols(
        engine,
        fold_config,
        training_start,
        train_end_date,
    )
    if not training_symbols:
        raise RuntimeError(f"{train_end_date} 没有符合条件的训练股票")
    bars = load_bars(
        engine,
        fold_config,
        training_start,
        train_end_date,
        symbols=training_symbols,
    )
    features = build_feature_frame(
        bars,
        workers=config.performance.workers,
        min_symbol_rows=config.training.min_symbol_rows,
    )
    metadata = train_models(features, fold_config)
    metadata["training_symbols"] = len(training_symbols)
    metadata["fold_train_end_date"] = train_end_date.isoformat()
    metadata["similarity"] = build_similarity_index(
        features,
        fold_directory,
        config.training.similarity_sample_limit,
        config.training.random_seed,
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return metadata


def _fixed_levels(
    close: float,
    take_pct: float,
    stop_pct: float,
    slippage_pct: float,
    tick: float,
) -> tuple[float, float, float]:
    take = round_to_tick(close * (1 + take_pct), tick)
    stop = round_to_tick(close * (1 - stop_pct), tick, "down")
    limit_price = round_to_tick(
        stop * (1 - slippage_pct),
        tick,
        "down",
    )
    return take, stop, limit_price


def _atr_levels(
    close: float,
    atr: float,
    take_multiplier: float,
    stop_multiplier: float,
    slippage_pct: float,
    tick: float,
) -> tuple[float, float, float]:
    take = round_to_tick(close + take_multiplier * atr, tick)
    stop = round_to_tick(close - stop_multiplier * atr, tick, "down")
    limit_price = round_to_tick(
        stop * (1 - slippage_pct),
        tick,
        "down",
    )
    return take, stop, limit_price


def _trade_record(
    *,
    run_id: str,
    strategy: str,
    train_end_date: date,
    row: pd.Series,
    take_profit: float,
    stop_trigger: float,
    stop_limit: float,
    confidence: float | None,
    risk_reward: float | None,
    model_version: str | None,
    stop_order_type: str,
) -> dict[str, Any]:
    simulation = simulate_next_day(
        close_price=float(row["raw_close"]),
        take_profit_price=take_profit,
        stop_trigger_price=stop_trigger,
        stop_limit_price=stop_limit,
        next_open=float(row["next_open"]),
        next_high=float(row["next_high"]),
        next_low=float(row["next_low"]),
        next_close=float(row["next_close"]),
        next_is_suspended=bool(row.get("next_is_suspended", False)),
        stop_order_type=stop_order_type,
    )
    return {
        "run_id": run_id,
        "strategy": strategy,
        "fold_train_end_date": train_end_date,
        "signal_date": row["trade_date"].date(),
        "execution_date": pd.Timestamp(row["execution_date"]).date(),
        "symbol": str(row["symbol"]),
        "close_price": float(row["raw_close"]),
        "take_profit_price": take_profit,
        "take_profit_enabled": int(take_profit is not None),
        "stop_trigger_price": stop_trigger,
        "stop_limit_price": stop_limit,
        "next_open": float(row["next_open"]),
        "next_high": float(row["next_high"]),
        "next_low": float(row["next_low"]),
        "next_close": float(row["next_close"]),
        "confidence": confidence,
        "risk_reward_ratio": risk_reward,
        "model_version": model_version,
        **simulation,
    }


def _hold_record(
    *,
    run_id: str,
    train_end_date: date,
    row: pd.Series,
) -> dict[str, Any]:
    close = float(row["raw_close"])
    next_close = float(row["next_close"])
    return {
        "run_id": run_id,
        "strategy": STRATEGY_HOLD,
        "fold_train_end_date": train_end_date,
        "signal_date": row["trade_date"].date(),
        "execution_date": pd.Timestamp(row["execution_date"]).date(),
        "symbol": str(row["symbol"]),
        "close_price": close,
        "take_profit_price": None,
        "take_profit_enabled": 0,
        "stop_trigger_price": close,
        "stop_limit_price": close,
        "next_open": float(row["next_open"]),
        "next_high": float(row["next_high"]),
        "next_low": float(row["next_low"]),
        "next_close": next_close,
        "exit_price": next_close,
        "outcome": "HOLD_CLOSE",
        "return_pct": float(next_close / close - 1),
        "confidence": None,
        "risk_reward_ratio": None,
        "stop_limit_unfilled": 0,
        "dual_hit": 0,
        "gap_stop": 0,
        "model_version": None,
    }


def run_backtest(
    engine: Any,
    config: AppConfig,
    *,
    start_date: date,
    end_date: date,
    max_symbols: int | None = None,
    stop_order_type: str | None = None,
    output_directory: str | Path = "output/backtests",
    write_database: bool = True,
    reuse_folds: bool = True,
) -> dict[str, Any]:
    if start_date >= end_date:
        raise ValueError("回测开始日期必须早于结束日期")
    max_symbols = max_symbols or config.backtest.max_symbols
    stop_order_type = stop_order_type or config.backtest.stop_order_type
    run_id = datetime.now(timezone.utc).strftime("bt_%Y%m%dT%H%M%SZ")
    output_root = Path(output_directory).resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    training_fingerprint = hashlib.sha1(
        json.dumps(
            asdict(config.training),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:12]
    model_root = (
        config.artifacts_directory
        / "backtests"
        / "folds"
        / training_fingerprint
    )

    symbols = _select_evaluation_symbols(
        engine,
        config,
        start_date,
        max_symbols,
    )
    evaluation = _prepare_evaluation_frame(
        engine,
        config,
        start_date,
        end_date,
        symbols,
    )
    print(
        f"回测股票 {len(symbols)} 支，有效信号 {len(evaluation):,} 条",
        flush=True,
    )

    all_trades: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for month, fold_frame in evaluation.groupby("month", sort=True):
        first_signal_date = fold_frame["trade_date"].min().date()
        train_end_date = _previous_trading_date(engine, first_signal_date)
        fold_directory = model_root / train_end_date.isoformat()
        print(
            f"开始回测月份 {month}：训练截止 {train_end_date}，"
            f"信号 {len(fold_frame):,} 条",
            flush=True,
        )
        metadata = _train_backtest_fold(
            engine,
            config,
            train_end_date,
            fold_directory,
            reuse=reuse_folds,
        )
        bundle = ModelBundle(fold_directory)
        similarity = SimilarityIndex(fold_directory)
        fold_frame = fold_frame.copy()
        fold_frame["predicted_high_return"] = bundle.predict(
            fold_frame,
            "high",
            config.recommendation.take_profit_quantile,
        )
        fold_frame["predicted_low_return"] = bundle.predict(
            fold_frame,
            "low",
            config.recommendation.stop_loss_quantile,
        )
        neighbors = similarity.query_batch(
            fold_frame,
            config.recommendation.similarity_top_k,
            sample_limit=config.backtest.similarity_query_sample_limit,
            batch_size=config.backtest.similarity_batch_size,
        )
        fold_frame = fold_frame.join(neighbors)

        for _, row in fold_frame.iterrows():
            close = float(row["raw_close"])
            atr = close * float(row["atr_14_pct"])
            hybrid = make_risk_decision(
                close=close,
                avg_cost=close,
                max_loss_pct=config.positions.default_max_loss_pct,
                current_stop=None,
                predicted_high_return=float(row["predicted_high_return"]),
                predicted_low_return=float(row["predicted_low_return"]),
                similar_high_return=float(row["median_high_return"]),
                similar_low_return=float(row["low_return_q20"]),
                atr_14=atr,
                similarity_distance=float(row["mean_distance"]),
                sample_count=int(row["sample_count"]),
                config=config.recommendation,
            )
            all_trades.append(
                _trade_record(
                    run_id=run_id,
                    strategy=STRATEGY_HYBRID,
                    train_end_date=train_end_date,
                    row=row,
                    take_profit=hybrid.take_profit_price,
                    stop_trigger=hybrid.stop_trigger_price,
                    stop_limit=hybrid.stop_limit_price,
                    confidence=hybrid.confidence,
                    risk_reward=hybrid.risk_reward_ratio,
                    model_version=bundle.version,
                    stop_order_type=stop_order_type,
                )
            )
            all_trades.append(
                _hold_record(
                    run_id=run_id,
                    train_end_date=train_end_date,
                    row=row,
                )
            )

            fixed = _fixed_levels(
                close,
                config.backtest.fixed_take_profit_pct,
                config.backtest.fixed_stop_loss_pct,
                config.recommendation.stop_limit_slippage_pct,
                config.recommendation.price_tick,
            )
            all_trades.append(
                _trade_record(
                    run_id=run_id,
                    strategy=STRATEGY_FIXED,
                    train_end_date=train_end_date,
                    row=row,
                    take_profit=fixed[0],
                    stop_trigger=fixed[1],
                    stop_limit=fixed[2],
                    confidence=None,
                    risk_reward=None,
                    model_version=None,
                    stop_order_type=stop_order_type,
                )
            )

            atr_levels = _atr_levels(
                close,
                atr,
                config.backtest.atr_take_profit_multiplier,
                config.backtest.atr_stop_loss_multiplier,
                config.recommendation.stop_limit_slippage_pct,
                config.recommendation.price_tick,
            )
            all_trades.append(
                _trade_record(
                    run_id=run_id,
                    strategy=STRATEGY_ATR,
                    train_end_date=train_end_date,
                    row=row,
                    take_profit=atr_levels[0],
                    stop_trigger=atr_levels[1],
                    stop_limit=atr_levels[2],
                    confidence=None,
                    risk_reward=None,
                    model_version=None,
                    stop_order_type=stop_order_type,
                )
            )
        folds.append(
            {
                "month": month,
                "train_end_date": train_end_date.isoformat(),
                "model_version": metadata["model_version"],
                "signal_count": int(len(fold_frame)),
            }
        )

    trades = pd.DataFrame(all_trades)
    metrics = {
        strategy: calculate_backtest_metrics(group)
        for strategy, group in trades.groupby("strategy")
    }
    hybrid = trades.loc[trades["strategy"] == STRATEGY_HYBRID].copy()
    if not hybrid.empty:
        hybrid["confidence_bucket"] = pd.cut(
            hybrid["confidence"],
            bins=[0.0, 0.50, 0.70, 0.85, 1.01],
            labels=["0-0.50", "0.50-0.70", "0.70-0.85", "0.85-1.00"],
            include_lowest=True,
        )
        confidence_metrics = {
            str(bucket): calculate_backtest_metrics(group)
            for bucket, group in hybrid.groupby(
                "confidence_bucket",
                observed=True,
            )
        }
    else:
        confidence_metrics = {}

    summary = {
        "run_id": run_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "retrain_frequency": "monthly",
        "execution_mode": "conservative",
        "stop_order_type": stop_order_type,
        "symbol_count": len(symbols),
        "signal_count": int(len(evaluation)),
        "fold_count": len(folds),
        "folds": folds,
        "metrics": metrics,
        "hybrid_confidence_metrics": confidence_metrics,
        "limitations": [
            "使用日线，双触发按止损先发生",
            "假定每个信号日按收盘价持有，非真实历史账户持仓",
            "股票池来自当前可用上市股票，存在幸存者偏差",
        ],
    }
    trades_path = output_root / "trades.csv"
    summary_path = output_root / "summary.json"
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if write_database:
        from .db import save_backtest_result

        save_backtest_result(
            engine,
            summary=summary,
            config_json={
                "training": asdict(config.training),
                "recommendation": asdict(config.recommendation),
                "backtest": asdict(config.backtest),
            },
            trades=trades,
        )
    summary["output_directory"] = str(output_root)
    return summary
