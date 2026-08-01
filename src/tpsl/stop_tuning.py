from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from .backtest import _select_evaluation_symbols, simulate_next_day
from .config import AppConfig
from .holding_backtest import (
    _maximum_drawdown,
    _predict_stops,
    _prepare_position_features,
    ratchet_stop,
)
from .risk import make_risk_decision, round_to_tick


_TUNING_WORKER_PATHS: list[dict[str, Any]] = []
_TUNING_WORKER_CONFIG: AppConfig | None = None


def build_parameter_grid(
    atr_multipliers: tuple[float, ...],
    limit_slippage_pcts: tuple[float, ...],
    minimum_stop_gap_pcts: tuple[float, ...],
    include_market_orders: bool,
) -> list[dict[str, float | str]]:
    grid = [
        {
            "stop_order_type": "limit",
            "atr_multiplier": float(atr),
            "limit_slippage_pct": float(slippage),
            "minimum_stop_gap_pct": float(stop_gap),
        }
        for atr in atr_multipliers
        for slippage in limit_slippage_pcts
        for stop_gap in minimum_stop_gap_pcts
    ]
    if include_market_orders:
        grid.extend(
            {
                "stop_order_type": "market",
                "atr_multiplier": float(atr),
                "limit_slippage_pct": 0.0,
                "minimum_stop_gap_pct": float(stop_gap),
            }
            for atr in atr_multipliers
            for stop_gap in minimum_stop_gap_pcts
        )
    return grid


def _entry_dates(
    engine: Any,
    start_date: date,
    end_date: date,
) -> list[date]:
    query = text(
        """
        SELECT MIN(trade_date) AS entry_date
        FROM stock_daily_bars
        WHERE trade_date BETWEEN :start_date AND :end_date
        GROUP BY YEAR(trade_date), MONTH(trade_date)
        ORDER BY entry_date
        """
    )
    with engine.connect() as connection:
        return [
            row[0]
            for row in connection.execute(
                query,
                {"start_date": start_date, "end_date": end_date},
            )
        ]


def _path_end_date(
    engine: Any,
    entry_date: date,
    holding_days: int,
) -> date:
    query = text(
        """
        SELECT trade_date
        FROM (
            SELECT DISTINCT trade_date
            FROM stock_daily_bars
            WHERE trade_date >= :entry_date
            ORDER BY trade_date
            LIMIT :row_count
        ) dates
        ORDER BY trade_date DESC
        LIMIT 1
        """
    )
    with engine.connect() as connection:
        dates = connection.execute(
            query,
            {
                "entry_date": entry_date,
                "row_count": holding_days + 1,
            },
        ).fetchall()
    if not dates:
        raise RuntimeError(f"{entry_date} 之后没有交易日")
    result = dates[0][0]
    count_query = text(
        """
        SELECT COUNT(*) FROM (
            SELECT DISTINCT trade_date
            FROM stock_daily_bars
            WHERE trade_date BETWEEN :entry_date AND :end_date
        ) dates
        """
    )
    with engine.connect() as connection:
        count = connection.execute(
            count_query,
            {"entry_date": entry_date, "end_date": result},
        ).scalar_one()
    if count < holding_days + 1:
        raise RuntimeError(
            f"{entry_date} 之后不足 {holding_days} 个完整交易日"
        )
    return result


def _build_paths(
    predicted: pd.DataFrame,
    symbols: list[str],
    entry_dates: list[date],
    holding_days_values: tuple[int, ...],
) -> list[dict[str, Any]]:
    paths: list[dict[str, Any]] = []
    grouped = {
        symbol: group.sort_values("trade_date").reset_index(drop=True)
        for symbol, group in predicted.groupby("symbol", sort=False)
        if symbol in symbols
    }
    for entry_date in entry_dates:
        for symbol in symbols:
            symbol_frame = grouped.get(symbol)
            if symbol_frame is None:
                continue
            entry_matches = symbol_frame.index[
                symbol_frame["trade_date"].dt.date == entry_date
            ].tolist()
            if not entry_matches:
                continue
            start_index = entry_matches[0]
            for holding_days in holding_days_values:
                rows = symbol_frame.iloc[
                    start_index : start_index + holding_days
                ].copy()
                if len(rows) != holding_days:
                    continue
                paths.append(
                    {
                        "entry_date": entry_date,
                        "symbol": symbol,
                        "holding_days": holding_days,
                        "entry_price": float(rows.iloc[0]["raw_close"]),
                        "rows": rows,
                    }
                )
    return paths


def _simulate_path(
    *,
    path: dict[str, Any],
    config: AppConfig,
    stop_order_type: str,
    atr_multiplier: float,
    limit_slippage_pct: float,
    minimum_stop_gap_pct: float,
    minimum_stop_gap_column: str | None = None,
) -> dict[str, Any]:
    entry_price = float(path["entry_price"])
    rows: pd.DataFrame = path["rows"]
    active_stop: float | None = None
    exit_price: float | None = None
    exit_date: date | None = None
    exit_outcome = "HOLD_TO_HORIZON"
    unfilled_count = 0
    stop_update_count = 0
    strategy_equity = [1.0]
    strategy_adverse = [0.0]
    hold_equity = [1.0] + (
        rows["next_close"].astype(float) / entry_price
    ).tolist()
    hold_adverse = [0.0] + (
        rows["next_low"].astype(float) / entry_price - 1.0
    ).tolist()
    used_stop_gaps: list[float] = []

    for row in rows.itertuples(index=False):
        close = float(row.raw_close)
        atr = close * float(row.atr_14_pct)
        row_stop_gap = float(minimum_stop_gap_pct)
        if minimum_stop_gap_column is not None:
            value = getattr(row, minimum_stop_gap_column, np.nan)
            if pd.notna(value):
                row_stop_gap = float(value)
        used_stop_gaps.append(row_stop_gap)
        recommendation = replace(
            config.recommendation,
            atr_stop_multiplier=atr_multiplier,
            stop_limit_slippage_pct=limit_slippage_pct,
            minimum_stop_gap_pct=row_stop_gap,
        )
        decision = make_risk_decision(
            close=close,
            avg_cost=entry_price,
            max_loss_pct=config.positions.default_max_loss_pct,
            current_stop=active_stop,
            predicted_high_return=float(row.predicted_high_return),
            predicted_low_return=float(row.predicted_low_return),
            similar_high_return=float(row.median_high_return),
            similar_low_return=float(row.low_return_q20),
            atr_14=atr,
            similarity_distance=float(row.mean_distance),
            sample_count=int(row.sample_count),
            config=recommendation,
        )
        previous_stop = active_stop
        active_stop = ratchet_stop(
            active_stop,
            decision.stop_trigger_price,
        )
        if (
            previous_stop is None
            or active_stop
            >= previous_stop + config.recommendation.price_tick
        ):
            stop_update_count += 1
        stop_limit = round_to_tick(
            active_stop * (1 - limit_slippage_pct),
            config.recommendation.price_tick,
            "down",
        )
        simulation = simulate_next_day(
            close_price=close,
            take_profit_price=None,
            stop_trigger_price=active_stop,
            stop_limit_price=stop_limit,
            next_open=float(row.next_open),
            next_high=float(row.next_high),
            next_low=float(row.next_low),
            next_close=float(row.next_close),
            next_is_suspended=bool(
                getattr(row, "next_is_suspended", False)
            ),
            stop_order_type=stop_order_type,
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
            exit_date = pd.Timestamp(row.execution_date).date()
            exit_outcome = event
            strategy_value = exit_price / entry_price
            strategy_adverse.append(strategy_value - 1.0)
        else:
            strategy_value = float(row.next_close) / entry_price
            strategy_adverse.append(
                float(row.next_low) / entry_price - 1.0
            )
        strategy_equity.append(strategy_value)
        if exited:
            break

    hold_return = float(rows.iloc[-1]["next_close"]) / entry_price - 1.0
    if exit_price is None:
        strategy_return = hold_return
    else:
        strategy_return = exit_price / entry_price - 1.0
        strategy_equity.append(strategy_equity[-1])

    excess_return = strategy_return - hold_return
    stopped = exit_price is not None
    return {
        "entry_date": path["entry_date"],
        "symbol": path["symbol"],
        "holding_days": int(path["holding_days"]),
        "stop_order_type": stop_order_type,
        "atr_multiplier": atr_multiplier,
        "limit_slippage_pct": limit_slippage_pct,
        "minimum_stop_gap_pct": minimum_stop_gap_pct,
        "average_minimum_stop_gap_pct": (
            float(np.mean(used_stop_gaps)) if used_stop_gaps else minimum_stop_gap_pct
        ),
        "min_minimum_stop_gap_pct": (
            float(np.min(used_stop_gaps)) if used_stop_gaps else minimum_stop_gap_pct
        ),
        "max_minimum_stop_gap_pct": (
            float(np.max(used_stop_gaps)) if used_stop_gaps else minimum_stop_gap_pct
        ),
        "entry_price": entry_price,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "exit_outcome": exit_outcome,
        "stopped": int(stopped),
        "strategy_return": strategy_return,
        "hold_return": hold_return,
        "excess_return": excess_return,
        "strategy_max_drawdown": _maximum_drawdown(strategy_equity),
        "hold_max_drawdown": _maximum_drawdown(hold_equity),
        "strategy_max_loss": float(min(strategy_adverse)),
        "hold_max_loss": float(min(hold_adverse)),
        "stop_update_count": stop_update_count,
        "stop_limit_unfilled_count": unfilled_count,
        "loss_avoided": int(stopped and excess_return > 0),
        "whipsaw": int(stopped and hold_return - strategy_return >= 0.02),
    }


def _initialize_tuning_worker(
    paths: list[dict[str, Any]],
    config: AppConfig,
) -> None:
    global _TUNING_WORKER_PATHS, _TUNING_WORKER_CONFIG
    _TUNING_WORKER_PATHS = paths
    _TUNING_WORKER_CONFIG = config


def _simulate_parameter_worker(
    parameter: dict[str, float | str],
) -> list[dict[str, Any]]:
    if _TUNING_WORKER_CONFIG is None:
        raise RuntimeError("止损调参工作进程未初始化")
    return [
        _simulate_path(
            path=path,
            config=_TUNING_WORKER_CONFIG,
            stop_order_type=str(parameter["stop_order_type"]),
            atr_multiplier=float(parameter["atr_multiplier"]),
            limit_slippage_pct=float(parameter["limit_slippage_pct"]),
            minimum_stop_gap_pct=float(
                parameter["minimum_stop_gap_pct"]
            ),
        )
        for path in _TUNING_WORKER_PATHS
    ]


def summarize_paths(paths: pd.DataFrame) -> dict[str, Any]:
    if paths.empty:
        raise ValueError("参数组合没有可汇总的模拟路径")
    stopped = paths["stopped"].astype(bool)
    return {
        "path_count": int(len(paths)),
        "stopped_rate": float(paths["stopped"].mean()),
        "stop_limit_unfilled_rate": float(
            (paths["stop_limit_unfilled_count"] > 0).mean()
        ),
        "average_strategy_return": float(
            paths["strategy_return"].mean()
        ),
        "average_hold_return": float(paths["hold_return"].mean()),
        "average_excess_return": float(paths["excess_return"].mean()),
        "p05_strategy_return": float(
            paths["strategy_return"].quantile(0.05)
        ),
        "median_strategy_return": float(
            paths["strategy_return"].median()
        ),
        "average_strategy_max_drawdown": float(
            paths["strategy_max_drawdown"].mean()
        ),
        "average_hold_max_drawdown": float(
            paths["hold_max_drawdown"].mean()
        ),
        "average_drawdown_reduction": float(
            (
                paths["strategy_max_drawdown"]
                - paths["hold_max_drawdown"]
            ).mean()
        ),
        "average_strategy_max_loss": float(
            paths["strategy_max_loss"].mean()
        ),
        "average_hold_max_loss": float(
            paths["hold_max_loss"].mean()
        ),
        "average_max_loss_reduction": float(
            (paths["strategy_max_loss"] - paths["hold_max_loss"]).mean()
        ),
        "loss_avoidance_rate": (
            float(paths.loc[stopped, "loss_avoided"].mean())
            if stopped.any()
            else 0.0
        ),
        "whipsaw_rate": (
            float(paths.loc[stopped, "whipsaw"].mean())
            if stopped.any()
            else 0.0
        ),
    }


def rank_results(results: pd.DataFrame) -> pd.DataFrame:
    ranked_parts: list[pd.DataFrame] = []
    for holding_days, group in results.groupby("holding_days", sort=True):
        ranked = group.copy()
        ranked["tail_rank"] = ranked["p05_strategy_return"].rank(
            ascending=False,
            method="min",
        )
        ranked["loss_rank"] = ranked["average_strategy_max_loss"].rank(
            ascending=False,
            method="min",
        )
        ranked["excess_rank"] = ranked["average_excess_return"].rank(
            ascending=False,
            method="min",
        )
        ranked["unfilled_rank"] = ranked[
            "stop_limit_unfilled_rate"
        ].rank(ascending=True, method="min")
        ranked["whipsaw_rank"] = ranked["whipsaw_rate"].rank(
            ascending=True,
            method="min",
        )
        ranked["stopped_rank"] = ranked["stopped_rate"].rank(
            ascending=True,
            method="min",
        )
        ranked["rank_score"] = (
            ranked["tail_rank"] * 2.0
            + ranked["loss_rank"] * 2.0
            + ranked["excess_rank"]
            + ranked["unfilled_rank"] * 0.5
            + ranked["whipsaw_rank"] * 0.5
            + ranked["stopped_rank"] * 0.5
        )
        ranked = ranked.sort_values(
            [
                "rank_score",
                "p05_strategy_return",
                "average_excess_return",
            ],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        ranked["rank_no"] = np.arange(1, len(ranked) + 1)
        ranked_parts.append(ranked)
    return pd.concat(ranked_parts, ignore_index=True)


def run_stop_tuning(
    engine: Any,
    config: AppConfig,
    *,
    entry_start_date: date,
    entry_end_date: date,
    max_symbols: int | None = None,
    output_directory: str | Path = "output/stop-tuning",
    write_database: bool = True,
) -> dict[str, Any]:
    if entry_start_date > entry_end_date:
        raise ValueError("模拟建仓开始日期不能晚于结束日期")
    max_symbols = max_symbols or config.stop_tuning.max_symbols
    entry_dates = _entry_dates(
        engine,
        entry_start_date,
        entry_end_date,
    )
    if not entry_dates:
        raise RuntimeError("模拟建仓范围没有交易日")
    max_holding_days = max(config.stop_tuning.holding_days)
    path_end_date = _path_end_date(
        engine,
        entry_dates[-1],
        max_holding_days,
    )
    symbols = _select_evaluation_symbols(
        engine,
        config,
        entry_start_date,
        max_symbols,
    )
    features = _prepare_position_features(
        engine,
        config,
        entry_dates[0],
        path_end_date,
        symbols,
    )
    signal_rows = features.loc[
        (features["trade_date"].dt.date >= entry_dates[0])
        & (features["trade_date"].dt.date < path_end_date)
    ].copy()
    predicted = _predict_stops(engine, config, signal_rows)
    paths = _build_paths(
        predicted,
        symbols,
        entry_dates,
        config.stop_tuning.holding_days,
    )
    if not paths:
        raise RuntimeError("没有构建出完整模拟持仓路径")

    grid = build_parameter_grid(
        config.stop_tuning.atr_multipliers,
        config.stop_tuning.limit_slippage_pcts,
        config.stop_tuning.minimum_stop_gap_pcts,
        config.stop_tuning.include_market_orders,
    )
    run_id = datetime.now(timezone.utc).strftime("st_%Y%m%dT%H%M%SZ")
    output_root = Path(output_directory).resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"止损参数扫描：{len(symbols)} 支股票，"
        f"{len(entry_dates)} 个建仓月，{len(paths)} 条基础路径，"
        f"{len(grid)} 组参数",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    workers = min(config.performance.workers, len(grid))
    if workers == 1:
        _initialize_tuning_worker(paths, config)
        for index, parameter in enumerate(grid, start=1):
            records.extend(_simulate_parameter_worker(parameter))
            print(
                f"扫描进度 {index}/{len(grid)}，工作进程 1",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_tuning_worker,
            initargs=(paths, config),
        ) as executor:
            futures = {
                executor.submit(
                    _simulate_parameter_worker,
                    parameter,
                ): parameter
                for parameter in grid
            }
            for completed, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                records.extend(future.result())
                print(
                    f"扫描进度 {completed}/{len(grid)}，"
                    f"工作进程 {workers}",
                    flush=True,
                )
    path_frame = pd.DataFrame(records)

    result_records: list[dict[str, Any]] = []
    parameter_columns = [
        "stop_order_type",
        "atr_multiplier",
        "limit_slippage_pct",
        "minimum_stop_gap_pct",
    ]
    for keys, group in path_frame.groupby(
        parameter_columns + ["holding_days"],
        sort=True,
    ):
        record = dict(zip(parameter_columns + ["holding_days"], keys))
        record.update(summarize_paths(group))
        result_records.append(record)
    for keys, group in path_frame.groupby(parameter_columns, sort=True):
        record = dict(zip(parameter_columns, keys))
        record["holding_days"] = 0
        record.update(summarize_paths(group))
        result_records.append(record)
    results = rank_results(pd.DataFrame(result_records))
    results.insert(0, "run_id", run_id)

    overall = results.loc[results["holding_days"] == 0].sort_values(
        "rank_no"
    )
    best = overall.iloc[0].to_dict()
    limit_overall = overall.loc[
        overall["stop_order_type"] == "limit"
    ]
    best_limit = (
        limit_overall.iloc[0].to_dict()
        if not limit_overall.empty
        else None
    )
    balanced_limit_candidates = limit_overall.loc[
        limit_overall["minimum_stop_gap_pct"]
        >= config.positions.default_max_loss_pct
    ].sort_values(
        [
            "average_strategy_return",
            "p05_strategy_return",
            "rank_no",
        ],
        ascending=[False, False, True],
    )
    balanced_limit = (
        balanced_limit_candidates.iloc[0].to_dict()
        if not balanced_limit_candidates.empty
        else best_limit
    )
    top_five = overall.head(5).to_dict("records")
    summary = {
        "run_id": run_id,
        "entry_start_date": entry_start_date.isoformat(),
        "entry_end_date": entry_end_date.isoformat(),
        "path_end_date": path_end_date.isoformat(),
        "symbol_count": len(symbols),
        "entry_dates": [value.isoformat() for value in entry_dates],
        "entry_count": len(symbols) * len(entry_dates),
        "base_path_count": len(paths),
        "combination_count": len(grid),
        "best_overall": best,
        "best_limit_order": best_limit,
        "balanced_limit_order": balanced_limit,
        "top_five_overall": top_five,
        "ranking_method": (
            "5%最差收益和最大亏损各2倍权重，"
            "超额收益1倍，未成交率、过早止损率和止损触发率各0.5倍"
        ),
        "limitations": [
            "每月首个交易日按收盘价模拟建仓",
            "每个历史月份使用月初前训练的模型",
            "使用日线模拟止损成交",
            "未计佣金、印花税和真实盘口滑点",
            "股票池来自当前可用上市股票，存在幸存者偏差",
        ],
    }
    paths_path = output_root / "paths.csv"
    results_path = output_root / "grid_results.csv"
    summary_path = output_root / "summary.json"
    path_frame.to_csv(paths_path, index=False, encoding="utf-8-sig")
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if write_database:
        from .db import save_stop_tuning_result

        save_stop_tuning_result(
            engine,
            summary=summary,
            config_json={
                "training": asdict(config.training),
                "recommendation": asdict(config.recommendation),
                "backtest": asdict(config.backtest),
                "stop_tuning": asdict(config.stop_tuning),
            },
            results=results,
        )
    summary["output_directory"] = str(output_root)
    return summary
