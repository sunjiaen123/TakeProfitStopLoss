from __future__ import annotations

import re
import json
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pandas as pd

from .config import AppConfig

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"非法 SQL 标识符：{value!r}")
    return f"`{value}`"


def create_engine(config: AppConfig) -> "Engine":
    try:
        from sqlalchemy import create_engine as sqlalchemy_create_engine
        from sqlalchemy.engine import make_url
    except ImportError as exc:
        raise RuntimeError("缺少 SQLAlchemy/PyMySQL，请先安装 requirements.txt") from exc

    target_url = make_url(config.database.url).set(database=config.database.name)
    return sqlalchemy_create_engine(
        target_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow,
    )


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


def _split_sql_statements(script: str) -> list[str]:
    statements: list[str] = []
    buffer: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(buffer).strip()
            statements.append(statement[:-1].strip())
            buffer = []
    if buffer:
        statements.append("\n".join(buffer).strip())
    return statements


def initialize_database(config: AppConfig) -> dict[str, object]:
    try:
        from sqlalchemy import create_engine as sqlalchemy_create_engine
        from sqlalchemy import inspect, text
        from sqlalchemy.engine import make_url
    except ImportError as exc:
        raise RuntimeError("缺少 SQLAlchemy/PyMySQL，请先安装 requirements.txt") from exc

    database_name = config.database.name
    _ident(database_name)
    base_url = make_url(config.database.url)
    server_url = base_url.set(database=None)
    server_engine = sqlalchemy_create_engine(
        server_url,
        pool_pre_ping=True,
        pool_recycle=1800,
    )
    try:
        with server_engine.begin() as connection:
            connection.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS {_ident(database_name)} "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
                )
            )
    finally:
        server_engine.dispose()

    target_engine = create_engine(config)
    script = _schema_path().read_text(encoding="utf-8")
    statements = _split_sql_statements(script)
    with target_engine.begin() as connection:
        for statement in statements:
            connection.exec_driver_sql(statement)

    inspector = inspect(target_engine)
    expected_tables = {
        "stock_master",
        config.database.tables["positions"],
        config.database.tables["bars"],
        config.database.tables["recommendations"],
        "tpsl_model_runs",
        "tpsl_backtest_runs",
        "tpsl_backtest_trades",
        "tpsl_holding_backtest_runs",
        "tpsl_holding_backtest_positions",
        "tpsl_holding_backtest_daily",
        "tpsl_stop_tuning_runs",
        "tpsl_stop_tuning_results",
    }
    actual_tables = set(inspector.get_table_names())
    missing = sorted(expected_tables.difference(actual_tables))
    target_engine.dispose()
    if missing:
        raise RuntimeError(f"数据库初始化后仍缺少表：{missing}")
    return {
        "database": database_name,
        "created_or_verified_tables": sorted(expected_tables),
        "statement_count": len(statements),
    }


def load_positions(engine: "Engine", config: AppConfig) -> pd.DataFrame:
    from sqlalchemy import text

    table = _ident(config.database.tables["positions"])
    columns = config.database.position_columns
    selected = [
        f"{_ident(columns['symbol'])} AS symbol",
        f"{_ident(columns['quantity'])} AS quantity",
        f"{_ident(columns['avg_cost'])} AS avg_cost",
    ]
    for alias in ("entry_date", "max_loss_pct", "current_stop"):
        source = columns.get(alias, "")
        selected.append(
            f"{_ident(source)} AS {alias}" if source else f"NULL AS {alias}"
        )
    status_column = _ident(columns["status"])
    sql = text(
        f"SELECT {', '.join(selected)} FROM {table} "
        f"WHERE {status_column} = :active_status "
        f"AND {_ident(columns['quantity'])} > 0"
    )
    frame = pd.read_sql(
        sql,
        engine,
        params={"active_status": config.positions.active_status},
    )
    if frame.empty:
        return frame
    frame["symbol"] = frame["symbol"].astype(str)
    for column in ("quantity", "avg_cost", "max_loss_pct", "current_stop"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame["quantity"].gt(0)
        & frame["avg_cost"].notna()
        & frame["avg_cost"].gt(0)
    ].copy()
    return frame


def load_bars(
    engine: "Engine",
    config: AppConfig,
    start_date: date,
    end_date: date,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    from sqlalchemy import bindparam, text

    table = _ident(config.database.tables["bars"])
    columns = config.database.bar_columns
    required_aliases = (
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    selected = [
        f"{_ident(columns[alias])} AS {alias}"
        for alias in required_aliases
    ]
    amount_source = columns.get("amount", "")
    industry_source = columns.get("industry", "")
    adj_factor_source = columns.get("adj_factor", "")
    suspended_source = columns.get("is_suspended", "")
    selected.append(
        f"{_ident(amount_source)} AS amount" if amount_source else "NULL AS amount"
    )
    selected.append(
        f"{_ident(industry_source)} AS industry"
        if industry_source
        else "'UNKNOWN' AS industry"
    )
    selected.append(
        f"{_ident(adj_factor_source)} AS adj_factor"
        if adj_factor_source
        else "1.0 AS adj_factor"
    )
    selected.append(
        f"{_ident(suspended_source)} AS is_suspended"
        if suspended_source
        else "0 AS is_suspended"
    )
    symbol_filter = ""
    if symbols:
        symbol_filter = f" AND {_ident(columns['symbol'])} IN :symbols"
    sql = text(
        f"SELECT {', '.join(selected)} FROM {table} "
        f"WHERE {_ident(columns['trade_date'])} BETWEEN :start_date AND :end_date"
        f"{symbol_filter} "
        f"ORDER BY {_ident(columns['symbol'])}, {_ident(columns['trade_date'])}"
    )
    if symbols:
        sql = sql.bindparams(bindparam("symbols", expanding=True))
    params: dict[str, object] = {
        "start_date": start_date,
        "end_date": end_date,
    }
    if symbols:
        params["symbols"] = list(symbols)
    frame = pd.read_sql(
        sql,
        engine,
        params=params,
    )
    if frame.empty:
        return frame
    frame["symbol"] = frame["symbol"].astype(str)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["industry"] = frame["industry"].fillna("UNKNOWN").astype(str)
    numeric = ["open", "high", "low", "close", "volume", "amount", "adj_factor"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["is_suspended"] = (
        pd.to_numeric(frame["is_suspended"], errors="coerce")
        .fillna(0)
        .astype("int8")
    )
    if frame["amount"].isna().all():
        frame["amount"] = frame["close"] * frame["volume"]
    frame = frame.dropna(
        subset=["trade_date", "open", "high", "low", "close"]
    )
    frame["volume"] = frame["volume"].fillna(0.0)
    frame["amount"] = frame["amount"].fillna(0.0)
    frame = frame.loc[
        frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
    ].copy()
    frame["adj_factor"] = frame["adj_factor"].fillna(1.0)
    frame.loc[frame["adj_factor"] <= 0, "adj_factor"] = 1.0
    for column in ("open", "high", "low", "close"):
        frame[f"raw_{column}"] = frame[column]
        frame[column] = frame[column] * frame["adj_factor"]
    float_columns = numeric + ["raw_open", "raw_high", "raw_low", "raw_close"]
    frame[float_columns] = frame[float_columns].astype("float32")
    return frame


def select_training_symbols(
    engine: "Engine",
    config: AppConfig,
    start_date: date,
    end_date: date,
) -> list[str]:
    from sqlalchemy import text

    table = _ident(config.database.tables["bars"])
    columns = config.database.bar_columns
    query = text(
        f"""
        SELECT {_ident(columns['symbol'])} AS symbol, COUNT(*) AS row_count
        FROM {table}
        WHERE {_ident(columns['trade_date'])} BETWEEN :start_date AND :end_date
        GROUP BY {_ident(columns['symbol'])}
        HAVING COUNT(*) >= :min_rows
        ORDER BY {_ident(columns['symbol'])}
        """
    )
    eligible = pd.read_sql(
        query,
        engine,
        params={
            "start_date": start_date,
            "end_date": end_date,
            "min_rows": config.training.min_symbol_rows,
        },
    )
    if eligible.empty:
        return []

    positions = load_positions(engine, config)
    forced = (
        set(positions["symbol"].astype(str))
        if not positions.empty
        else set()
    )
    eligible_symbols = eligible["symbol"].astype(str)
    forced = forced.intersection(set(eligible_symbols))
    limit = config.training.max_training_symbols
    if len(eligible) <= limit:
        return sorted(eligible_symbols.tolist())

    remaining_slots = max(0, limit - len(forced))
    candidates = eligible.loc[~eligible_symbols.isin(forced)]
    sampled = candidates.sample(
        n=min(remaining_slots, len(candidates)),
        random_state=config.training.random_seed,
    )["symbol"].astype(str)
    return sorted(forced.union(set(sampled)))


def upsert_recommendations(
    engine: "Engine",
    config: AppConfig,
    recommendations: pd.DataFrame,
) -> int:
    if recommendations.empty:
        return 0

    from sqlalchemy import text

    table = _ident(config.database.tables["recommendations"])
    columns = list(recommendations.columns)
    for column in columns:
        _ident(column)
    column_sql = ", ".join(_ident(column) for column in columns)
    values_sql = ", ".join(f":{column}" for column in columns)
    insert_sql = text(
        f"INSERT INTO {table} ({column_sql}) VALUES ({values_sql})"
    )
    records = recommendations.where(pd.notna(recommendations), None).to_dict("records")
    with engine.begin() as connection:
        keys = {
            (
                record["as_of_date"],
                record["symbol"],
                record["model_version"],
            )
            for record in records
        }
        delete_sql = text(
            f"DELETE FROM {table} "
            "WHERE `as_of_date` = :as_of_date "
            "AND `symbol` = :symbol "
            "AND `model_version` = :model_version"
        )
        connection.execute(
            delete_sql,
            [
                {
                    "as_of_date": as_of_date,
                    "symbol": symbol,
                    "model_version": model_version,
                }
                for as_of_date, symbol, model_version in keys
            ],
        )
        connection.execute(insert_sql, records)
    return len(records)


def record_model_run(
    engine: "Engine",
    *,
    model_version: str,
    train_end_date: date,
    status: str,
    device: str,
    training_rows: int,
    validation_rows: int,
    metrics: dict[str, object],
    error_message: str | None = None,
) -> None:
    from sqlalchemy import text

    delete_sql = text(
        "DELETE FROM tpsl_model_runs WHERE model_version = :model_version"
    )
    insert_sql = text(
        """
        INSERT INTO tpsl_model_runs (
            model_version, train_end_date, status, device,
            training_rows, validation_rows, metrics_json,
            error_message, finished_at
        ) VALUES (
            :model_version, :train_end_date, :status, :device,
            :training_rows, :validation_rows, :metrics_json,
            :error_message, CURRENT_TIMESTAMP
        )
        """
    )
    record = {
        "model_version": model_version,
        "train_end_date": train_end_date,
        "status": status,
        "device": device,
        "training_rows": training_rows,
        "validation_rows": validation_rows,
        "metrics_json": json.dumps(metrics, ensure_ascii=False),
        "error_message": error_message,
    }
    with engine.begin() as connection:
        connection.execute(delete_sql, {"model_version": model_version})
        connection.execute(insert_sql, record)


def save_backtest_result(
    engine: "Engine",
    *,
    summary: dict[str, object],
    config_json: dict[str, object],
    trades: pd.DataFrame,
    batch_size: int = 2000,
) -> None:
    from sqlalchemy import text

    run_id = str(summary["run_id"])
    run_record = {
        "run_id": run_id,
        "start_date": summary["start_date"],
        "end_date": summary["end_date"],
        "retrain_frequency": summary["retrain_frequency"],
        "execution_mode": summary["execution_mode"],
        "stop_order_type": summary["stop_order_type"],
        "symbol_count": summary["symbol_count"],
        "fold_count": summary["fold_count"],
        "status": "SUCCESS",
        "config_json": json.dumps(config_json, ensure_ascii=False),
        "metrics_json": json.dumps(
            {
                "metrics": summary["metrics"],
                "hybrid_confidence_metrics": summary[
                    "hybrid_confidence_metrics"
                ],
                "limitations": summary["limitations"],
            },
            ensure_ascii=False,
        ),
    }
    run_sql = text(
        """
        INSERT INTO tpsl_backtest_runs (
            run_id, start_date, end_date, retrain_frequency,
            execution_mode, stop_order_type, symbol_count, fold_count,
            status, config_json, metrics_json, finished_at
        ) VALUES (
            :run_id, :start_date, :end_date, :retrain_frequency,
            :execution_mode, :stop_order_type, :symbol_count, :fold_count,
            :status, :config_json, :metrics_json, CURRENT_TIMESTAMP
        )
        """
    )
    trade_columns = [
        "run_id",
        "strategy",
        "fold_train_end_date",
        "signal_date",
        "execution_date",
        "symbol",
        "close_price",
        "take_profit_price",
        "take_profit_enabled",
        "stop_trigger_price",
        "stop_limit_price",
        "next_open",
        "next_high",
        "next_low",
        "next_close",
        "exit_price",
        "outcome",
        "return_pct",
        "confidence",
        "risk_reward_ratio",
        "stop_limit_unfilled",
        "dual_hit",
        "gap_stop",
        "model_version",
    ]
    trade_sql = text(
        f"""
        INSERT INTO tpsl_backtest_trades (
            {", ".join(f"`{column}`" for column in trade_columns)}
        ) VALUES (
            {", ".join(f":{column}" for column in trade_columns)}
        )
        """
    )
    clean = trades[trade_columns].astype(object).where(
        pd.notna(trades[trade_columns]),
        None,
    )
    records = clean.to_dict("records")
    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM tpsl_backtest_trades WHERE run_id = :run_id"),
            {"run_id": run_id},
        )
        connection.execute(
            text("DELETE FROM tpsl_backtest_runs WHERE run_id = :run_id"),
            {"run_id": run_id},
        )
        connection.execute(run_sql, run_record)
        for start in range(0, len(records), batch_size):
            connection.execute(trade_sql, records[start : start + batch_size])


def save_holding_backtest_result(
    engine: "Engine",
    *,
    result: dict[str, object],
    config_json: dict[str, object],
    positions: pd.DataFrame,
    daily: pd.DataFrame,
    batch_size: int = 2000,
) -> None:
    from sqlalchemy import text

    run_id = str(result["run_id"])
    run_sql = text(
        """
        INSERT INTO tpsl_holding_backtest_runs (
            run_id, start_date, end_date, stop_order_type,
            position_count, status, config_json, metrics_json, finished_at
        ) VALUES (
            :run_id, :start_date, :end_date, :stop_order_type,
            :position_count, 'SUCCESS', :config_json, :metrics_json,
            CURRENT_TIMESTAMP
        )
        """
    )
    position_columns = [
        "run_id",
        "symbol",
        "entry_date",
        "avg_cost",
        "quantity",
        "exit_date",
        "exit_price",
        "exit_outcome",
        "final_stop_price",
        "strategy_return",
        "hold_return",
        "excess_return",
        "strategy_max_drawdown",
        "hold_max_drawdown",
        "strategy_max_loss_from_cost",
        "hold_max_loss_from_cost",
        "stop_update_count",
        "stop_limit_unfilled_count",
        "trading_days",
    ]
    daily_columns = [
        "run_id",
        "symbol",
        "signal_date",
        "execution_date",
        "close_price",
        "proposed_stop_price",
        "active_stop_price",
        "stop_limit_price",
        "next_open",
        "next_high",
        "next_low",
        "next_close",
        "event",
        "strategy_equity",
        "hold_equity",
        "confidence",
        "model_version",
    ]
    position_sql = text(
        f"""
        INSERT INTO tpsl_holding_backtest_positions (
            {", ".join(f"`{column}`" for column in position_columns)}
        ) VALUES (
            {", ".join(f":{column}" for column in position_columns)}
        )
        """
    )
    daily_sql = text(
        f"""
        INSERT INTO tpsl_holding_backtest_daily (
            {", ".join(f"`{column}`" for column in daily_columns)}
        ) VALUES (
            {", ".join(f":{column}" for column in daily_columns)}
        )
        """
    )
    position_records = positions[position_columns].astype(object).where(
        pd.notna(positions[position_columns]),
        None,
    ).to_dict("records")
    daily_records = (
        daily[daily_columns]
        .astype(object)
        .where(pd.notna(daily[daily_columns]), None)
        .to_dict("records")
        if not daily.empty
        else []
    )
    run_record = {
        "run_id": run_id,
        "start_date": result["start_date"],
        "end_date": result["end_date"],
        "stop_order_type": result["stop_order_type"],
        "position_count": len(position_records),
        "config_json": json.dumps(config_json, ensure_ascii=False),
        "metrics_json": json.dumps(
            {
                "metrics": result["metrics"],
                "limitations": result["limitations"],
            },
            ensure_ascii=False,
        ),
    }
    with engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM tpsl_holding_backtest_daily "
                "WHERE run_id = :run_id"
            ),
            {"run_id": run_id},
        )
        connection.execute(
            text(
                "DELETE FROM tpsl_holding_backtest_positions "
                "WHERE run_id = :run_id"
            ),
            {"run_id": run_id},
        )
        connection.execute(
            text(
                "DELETE FROM tpsl_holding_backtest_runs "
                "WHERE run_id = :run_id"
            ),
            {"run_id": run_id},
        )
        connection.execute(run_sql, run_record)
        if position_records:
            connection.execute(position_sql, position_records)
        for start in range(0, len(daily_records), batch_size):
            connection.execute(
                daily_sql,
                daily_records[start : start + batch_size],
            )


def save_stop_tuning_result(
    engine: "Engine",
    *,
    summary: dict[str, object],
    config_json: dict[str, object],
    results: pd.DataFrame,
) -> None:
    from sqlalchemy import text

    run_id = str(summary["run_id"])
    run_sql = text(
        """
        INSERT INTO tpsl_stop_tuning_runs (
            run_id, entry_start_date, entry_end_date, path_end_date,
            symbol_count, entry_count, combination_count, status,
            config_json, metrics_json, finished_at
        ) VALUES (
            :run_id, :entry_start_date, :entry_end_date, :path_end_date,
            :symbol_count, :entry_count, :combination_count, 'SUCCESS',
            :config_json, :metrics_json, CURRENT_TIMESTAMP
        )
        """
    )
    result_columns = [
        "run_id",
        "holding_days",
        "stop_order_type",
        "atr_multiplier",
        "limit_slippage_pct",
        "minimum_stop_gap_pct",
        "path_count",
        "stopped_rate",
        "stop_limit_unfilled_rate",
        "average_strategy_return",
        "average_hold_return",
        "average_excess_return",
        "p05_strategy_return",
        "median_strategy_return",
        "average_strategy_max_drawdown",
        "average_hold_max_drawdown",
        "average_drawdown_reduction",
        "average_strategy_max_loss",
        "average_hold_max_loss",
        "average_max_loss_reduction",
        "loss_avoidance_rate",
        "whipsaw_rate",
        "rank_score",
        "rank_no",
    ]
    result_sql = text(
        f"""
        INSERT INTO tpsl_stop_tuning_results (
            {", ".join(f"`{column}`" for column in result_columns)}
        ) VALUES (
            {", ".join(f":{column}" for column in result_columns)}
        )
        """
    )
    records = (
        results[result_columns]
        .astype(object)
        .where(pd.notna(results[result_columns]), None)
        .to_dict("records")
    )
    run_record = {
        "run_id": run_id,
        "entry_start_date": summary["entry_start_date"],
        "entry_end_date": summary["entry_end_date"],
        "path_end_date": summary["path_end_date"],
        "symbol_count": summary["symbol_count"],
        "entry_count": summary["entry_count"],
        "combination_count": summary["combination_count"],
        "config_json": json.dumps(config_json, ensure_ascii=False),
        "metrics_json": json.dumps(
            {
                "best_overall": summary["best_overall"],
                "top_five_overall": summary["top_five_overall"],
                "ranking_method": summary["ranking_method"],
                "limitations": summary["limitations"],
            },
            ensure_ascii=False,
            default=str,
        ),
    }
    with engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM tpsl_stop_tuning_results "
                "WHERE run_id = :run_id"
            ),
            {"run_id": run_id},
        )
        connection.execute(
            text(
                "DELETE FROM tpsl_stop_tuning_runs "
                "WHERE run_id = :run_id"
            ),
            {"run_id": run_id},
        )
        connection.execute(run_sql, run_record)
        connection.execute(result_sql, records)
