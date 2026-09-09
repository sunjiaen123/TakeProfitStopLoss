from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .chart_exit import add_chart_exit_indicators, recommend_chart_exit
from .config import AppConfig
from .db import (
    _ident,
    load_bars,
    load_positions,
    record_model_run,
    select_training_symbols,
    upsert_recommendations,
)
from .features import FEATURE_COLUMNS, build_feature_frame
from .model import ModelBundle, train_models
from .risk import make_risk_decision
from .risk_fit import RiskStopModel, build_risk_feature_frame, risk_model_directory
from .similarity import SimilarityIndex, build_similarity_index
from .volatility_stop import (
    VOLATILITY_STOP_GAP_COLUMN,
    add_volatility_stop_gaps,
    load_volatility_stop_state,
)


S3_OPERATIONAL_COLUMNS = (
    "as_of_date",
    "symbol",
    "close_price",
    "avg_cost",
    "chart_exit_action",
    "chart_exit_reason",
    "chart_exit_stop_trigger_price",
    "chart_exit_stop_limit_price",
)


def train_pipeline(
    engine: Any,
    config: AppConfig,
    end_date: date,
) -> dict[str, Any]:
    start_date = end_date - timedelta(days=config.training.lookback_days * 2)
    symbols = select_training_symbols(engine, config, start_date, end_date)
    if not symbols:
        raise RuntimeError("没有符合最小历史长度要求的训练股票")
    print(
        f"训练股票抽样完成：{len(symbols)} 支，"
        f"日期范围 {start_date} 至 {end_date}",
        flush=True,
    )
    bars = load_bars(
        engine,
        config,
        start_date,
        end_date,
        symbols=symbols,
    )
    if bars.empty:
        raise RuntimeError(f"{start_date} 至 {end_date} 没有读取到日线数据")
    print(f"日线读取完成：{len(bars):,} 行，开始计算特征", flush=True)
    features = build_feature_frame(
        bars,
        workers=config.performance.workers,
        min_symbol_rows=config.training.min_symbol_rows,
    )
    print(f"特征计算完成：{len(features):,} 行，开始训练模型", flush=True)
    metadata = train_models(features, config)
    metadata["training_symbols"] = len(symbols)
    similarity = build_similarity_index(
        features,
        config.artifacts_directory,
        config.training.similarity_sample_limit,
        config.training.random_seed,
    )
    print(
        f"相似形态索引完成：{similarity['sample_count']:,} 个样本",
        flush=True,
    )
    metadata["similarity"] = similarity
    (config.artifacts_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    record_model_run(
        engine,
        model_version=str(metadata["model_version"]),
        train_end_date=end_date,
        status="SUCCESS",
        device=",".join(str(value) for value in metadata["devices"]),
        training_rows=int(metadata["training_rows"]),
        validation_rows=int(metadata["validation_rows"]),
        metrics=dict(metadata["metrics"]),
    )
    return metadata


def _latest_rows_for_positions(
    features: pd.DataFrame,
    positions: pd.DataFrame,
    as_of_date: date,
) -> pd.DataFrame:
    target_date = pd.Timestamp(as_of_date)
    current = features.loc[
        (features["trade_date"].dt.normalize() == target_date)
        & features["symbol"].isin(positions["symbol"])
    ].copy()
    if current.empty:
        return current
    return current.merge(positions, on="symbol", how="inner", suffixes=("", "_position"))


def latest_complete_position_bar_date(
    engine: Any,
    config: AppConfig,
    positions: pd.DataFrame | None = None,
    end_date: date | None = None,
) -> date | None:
    positions = load_positions(engine, config) if positions is None else positions
    if positions.empty:
        return None
    symbols = sorted(set(positions["symbol"].astype(str)))
    if not symbols:
        return None

    from sqlalchemy import bindparam, text

    table = _ident(config.database.tables["bars"])
    symbol_column = _ident(config.database.bar_columns["symbol"])
    date_column = _ident(config.database.bar_columns["trade_date"])
    where_parts = [f"{symbol_column} IN :symbols"]
    params: dict[str, Any] = {
        "symbols": symbols,
        "symbol_count": len(symbols),
    }
    if end_date is not None:
        where_parts.append(f"{date_column} <= :end_date")
        params["end_date"] = end_date
    sql = text(
        f"""
        SELECT {date_column} AS trade_date
        FROM {table}
        WHERE {" AND ".join(where_parts)}
        GROUP BY {date_column}
        HAVING COUNT(DISTINCT {symbol_column}) = :symbol_count
        ORDER BY {date_column} DESC
        LIMIT 1
        """
    ).bindparams(bindparam("symbols", expanding=True))
    with engine.connect() as connection:
        row = connection.execute(sql, params).fetchone()
    if row is None:
        return None
    return pd.Timestamp(row[0]).date()


def resolve_recommend_as_of_date(
    engine: Any,
    config: AppConfig,
    requested: date | str,
) -> date:
    if isinstance(requested, date):
        return requested
    if str(requested).strip().lower() != "latest":
        raise ValueError("as-of-date 只能是 YYYY-MM-DD 或 latest")
    latest_date = latest_complete_position_bar_date(engine, config)
    if latest_date is None:
        raise RuntimeError("没有找到所有有效持仓都具备行情的日期，请先同步持仓日线")
    return latest_date


def _market_data_error_message(
    *,
    as_of_date: date,
    latest_date: date | None,
    missing_symbols: set[str],
) -> str:
    missing = ""
    if missing_symbols:
        missing = "缺失股票：" + "、".join(sorted(missing_symbols)) + "。"
    latest = (
        f"当前所有持仓的最新完整行情日是 {latest_date}。"
        if latest_date is not None
        else "当前没有找到所有持仓都具备行情的日期。"
    )
    return (
        f"{as_of_date} 没有找到持仓股票的完整行情。"
        f"{missing}{latest}"
        "如果要用库里最新可用行情，请运行 --as-of-date latest；"
        f"如果要用 {as_of_date}，请先执行 sync-data --scope positions "
        f"--end-date {as_of_date}。"
    )


def recommend_pipeline(
    engine: Any,
    config: AppConfig,
    as_of_date: date,
    write_database: bool = True,
) -> pd.DataFrame:
    positions = load_positions(engine, config)
    if positions.empty:
        return pd.DataFrame()

    start_date = as_of_date - timedelta(days=220)
    bars = load_bars(engine, config, start_date, as_of_date)
    if bars.empty:
        raise RuntimeError(f"{start_date} 至 {as_of_date} 没有读取到日线数据")
    features = build_feature_frame(
        bars,
        workers=config.performance.workers,
        min_symbol_rows=25,
    )
    volatility_state: dict[str, Any] | None = None
    volatility_error: str | None = None
    if config.volatility_stop.enabled:
        try:
            volatility_state = load_volatility_stop_state(config)
            features = add_volatility_stop_gaps(
                features,
                config,
                k=float(volatility_state["k"]),
            )
        except Exception as exc:
            volatility_state = None
            volatility_error = str(exc)
    chart_histories: dict[str, pd.DataFrame] = {}
    if config.chart_exit.enabled:
        features = add_chart_exit_indicators(features, config)
        chart_histories = {
            str(symbol): group.copy()
            for symbol, group in features.groupby("symbol", sort=False)
        }
    current = _latest_rows_for_positions(features, positions, as_of_date)
    position_symbols = set(positions["symbol"].astype(str))
    current_symbols = (
        set(current["symbol"].astype(str)) if not current.empty else set()
    )
    if current.empty or current_symbols != position_symbols:
        latest_date = latest_complete_position_bar_date(
            engine,
            config,
            positions,
            as_of_date,
        )
        raise RuntimeError(
            _market_data_error_message(
                as_of_date=as_of_date,
                latest_date=latest_date,
                missing_symbols=position_symbols - current_symbols,
            )
        )
    current = current.dropna(subset=FEATURE_COLUMNS + ["atr_14"])
    if current.empty:
        raise RuntimeError("持仓股票因历史长度不足或数据缺失，无法生成完整特征")

    # S3 is a complete exit strategy, not an auxiliary annotation on the
    # legacy next-day prediction.  When it is enabled, keep the public output
    # and the canonical stop prices exclusively on the S3 path.  This also
    # avoids loading the old model/similarity artifacts for an S3 run.
    if config.chart_exit.enabled:
        results: list[dict[str, Any]] = []
        database_results: list[dict[str, Any]] = []
        for _, row in current.iterrows():
            chart_decision = recommend_chart_exit(
                chart_histories.get(str(row["symbol"]), pd.DataFrame()),
                row,
                as_of_date,
                config,
            )
            model_version = f"chart_exit:{chart_decision.profile}"
            public_record = {
                "as_of_date": as_of_date,
                "symbol": str(row["symbol"]),
                "close_price": float(row.get("raw_close", row["close"])),
                "avg_cost": float(row["avg_cost"]),
                "model_version": model_version,
                "strategy_profile": chart_decision.profile,
                "chart_exit_action": chart_decision.action,
                "chart_exit_reason": chart_decision.reason,
                "chart_exit_stop_trigger_price": chart_decision.stop_trigger_price,
                "chart_exit_stop_limit_price": chart_decision.stop_limit_price,
                "chart_exit_position_scale": chart_decision.position_scale,
                "chart_exit_trend_active": (
                    int(bool(chart_decision.trend_active))
                    if chart_decision.trend_active is not None
                    else None
                ),
                "chart_exit_trade_days": chart_decision.trade_days,
                "chart_exit_held_peak_close": chart_decision.held_peak_close,
                "chart_exit_initial_stop_price": chart_decision.initial_stop_price,
                "chart_exit_progress_price": chart_decision.progress_price,
                "chart_exit_profit_floor_price": chart_decision.profit_floor_price,
                "chart_exit_entry_swing_low": chart_decision.entry_swing_low,
                "chart_exit_ma_fast_price": chart_decision.ma_fast_price,
                "chart_exit_ma_trend_price": chart_decision.ma_trend_price,
                "chart_exit_ma_long_price": chart_decision.ma_long_price,
                "chart_exit_ma_trend_slope": chart_decision.ma_trend_slope,
                "chart_exit_recent_swing_low": chart_decision.recent_swing_low,
                "chart_exit_diagnostic": chart_decision.diagnostic,
            }
            results.append(public_record)

            # Keep old NOT NULL database columns populated for installations
            # created with the original schema.  These are compatibility
            # placeholders, not legacy strategy calculations.  Canonical stop
            # columns contain the same S3 prices as chart_exit_*.
            database_results.append(
                {
                    **public_record,
                    "take_profit_price": None,
                    "take_profit_enabled": 0,
                    "take_profit_reason": "S3 不设置固定止盈单",
                    "stop_trigger_price": chart_decision.stop_trigger_price,
                    "stop_limit_price": chart_decision.stop_limit_price,
                    "dynamic_stop_gap_pct": None,
                    "predicted_high_return": 0.0,
                    "predicted_low_return": 0.0,
                    "similar_high_return": None,
                    "similar_low_return": None,
                    "risk_reward_ratio": None,
                    "confidence": 0.0,
                    "sample_count": 0,
                    "risk_model_version": None,
                    "chart_exit_enabled": 1,
                    "reason": (
                        f"S3={chart_decision.profile}；"
                        f"action={chart_decision.action}；"
                        f"reason={chart_decision.reason}"
                    ),
                }
            )

        output = pd.DataFrame(results)
        if write_database:
            upsert_recommendations(engine, config, pd.DataFrame(database_results))
        return output

    bundle = ModelBundle(config.artifacts_directory)
    similarity = SimilarityIndex(config.artifacts_directory)
    risk_model: RiskStopModel | None = None
    risk_model_error: str | None = None
    if (
        not config.volatility_stop.enabled
        and config.risk_fit.enabled
        and RiskStopModel.exists(config)
    ):
        try:
            risk_model = RiskStopModel(risk_model_directory(config))
        except Exception as exc:
            risk_model_error = str(exc)
    high_quantile = config.recommendation.take_profit_quantile
    low_quantile = config.recommendation.stop_loss_quantile
    current["predicted_high_return"] = bundle.predict(
        current,
        "high",
        high_quantile,
    )
    current["predicted_low_return"] = bundle.predict(
        current,
        "low",
        low_quantile,
    )

    results: list[dict[str, Any]] = []
    for _, row in current.iterrows():
        neighbors = similarity.query(
            row,
            config.recommendation.similarity_top_k,
        )
        max_loss = row.get("max_loss_pct")
        if pd.isna(max_loss) or not 0 < float(max_loss) < 1:
            max_loss = config.positions.default_max_loss_pct
        current_stop = row.get("current_stop")
        if pd.isna(current_stop):
            current_stop = None

        decision_config = config.recommendation
        dynamic_stop_gap: float | None = None
        risk_model_version: str | None = None
        local_risk_model_error = risk_model_error
        gap_source: str | None = None
        if volatility_state is not None:
            dynamic_stop_gap = float(row[VOLATILITY_STOP_GAP_COLUMN])
            risk_model_version = f"volatility:{volatility_state['version']}"
            gap_source = "volatility"
            decision_config = replace(
                config.recommendation,
                atr_stop_multiplier=0.0,
                stop_limit_slippage_pct=config.volatility_stop.limit_slippage_pct,
                minimum_stop_gap_pct=dynamic_stop_gap,
            )
        elif risk_model is not None:
            try:
                risk_frame = build_risk_feature_frame(row, neighbors)
                dynamic_stop_gap = float(risk_model.predict_gap(risk_frame)[0])
                risk_model_version = risk_model.version
                gap_source = "risk_model"
                decision_config = replace(
                    config.recommendation,
                    minimum_stop_gap_pct=dynamic_stop_gap,
                )
            except Exception as exc:
                local_risk_model_error = str(exc)
                dynamic_stop_gap = None
                risk_model_version = None

        decision = make_risk_decision(
            close=float(row.get("raw_close", row["close"])),
            avg_cost=float(row["avg_cost"]),
            max_loss_pct=float(max_loss),
            current_stop=None if current_stop is None else float(current_stop),
            predicted_high_return=float(row["predicted_high_return"]),
            predicted_low_return=float(row["predicted_low_return"]),
            similar_high_return=float(neighbors["median_high_return"]),
            similar_low_return=float(neighbors["low_return_q20"]),
            atr_14=float(row.get("raw_close", row["close"]))
            * float(row["atr_14_pct"]),
            similarity_distance=float(neighbors["mean_distance"]),
            sample_count=int(neighbors["sample_count"]),
            config=decision_config,
        )
        reason = decision.reason
        if dynamic_stop_gap is not None:
            if gap_source == "volatility":
                reason += (
                    f"；波动止损距离={dynamic_stop_gap:.2%}，"
                    f"k={float(volatility_state['k']):.2f}，"
                    f"风控版本={risk_model_version}；ATR stop 已关闭避免重复计算波动"
                )
            else:
                reason += (
                    f"；动态止损距离={dynamic_stop_gap:.2%}，"
                    f"风控模型={risk_model_version}"
                )
        elif volatility_error:
            reason += (
                "；波动止损未使用："
                f"{volatility_error}，使用固定止损距离="
                f"{config.recommendation.minimum_stop_gap_pct:.2%}"
            )
        elif local_risk_model_error:
            reason += (
                "；动态风控模型未使用："
                f"{local_risk_model_error}，使用固定止损距离="
                f"{config.recommendation.minimum_stop_gap_pct:.2%}"
            )
        else:
            reason += (
                "；未配置动态风控模型，使用固定止损距离="
                f"{config.recommendation.minimum_stop_gap_pct:.2%}"
            )
        chart_decision = None
        if config.chart_exit.enabled:
            chart_decision = recommend_chart_exit(
                chart_histories.get(str(row["symbol"]), pd.DataFrame()),
                row,
                as_of_date,
                config,
            )
            reason += (
                f"; chart_exit={chart_decision.profile}"
                f" action={chart_decision.action}"
                f" reason={chart_decision.reason}"
                f" position_scale={chart_decision.position_scale:.2f}"
            )
            if chart_decision.diagnostic:
                reason += f" diagnostic={chart_decision.diagnostic}"
        results.append(
            {
                "as_of_date": as_of_date,
                "symbol": str(row["symbol"]),
                "close_price": float(row.get("raw_close", row["close"])),
                "avg_cost": float(row["avg_cost"]),
                "take_profit_price": decision.take_profit_price,
                "take_profit_enabled": int(decision.take_profit_enabled),
                "take_profit_reason": decision.take_profit_reason,
                "stop_trigger_price": decision.stop_trigger_price,
                "stop_limit_price": decision.stop_limit_price,
                "dynamic_stop_gap_pct": dynamic_stop_gap,
                "predicted_high_return": float(row["predicted_high_return"]),
                "predicted_low_return": float(row["predicted_low_return"]),
                "similar_high_return": float(neighbors["median_high_return"]),
                "similar_low_return": float(neighbors["low_return_q20"]),
                "risk_reward_ratio": decision.risk_reward_ratio,
                "confidence": decision.confidence,
                "sample_count": int(neighbors["sample_count"]),
                "model_version": bundle.version,
                "risk_model_version": risk_model_version,
                "strategy_profile": (
                    chart_decision.profile if chart_decision else "B_production"
                ),
                "chart_exit_enabled": int(config.chart_exit.enabled),
                "chart_exit_action": (
                    chart_decision.action if chart_decision else "DISABLED"
                ),
                "chart_exit_reason": chart_decision.reason if chart_decision else None,
                "chart_exit_stop_trigger_price": (
                    chart_decision.stop_trigger_price if chart_decision else None
                ),
                "chart_exit_stop_limit_price": (
                    chart_decision.stop_limit_price if chart_decision else None
                ),
                "chart_exit_position_scale": (
                    chart_decision.position_scale if chart_decision else None
                ),
                "chart_exit_held_peak_close": (
                    chart_decision.held_peak_close if chart_decision else None
                ),
                "chart_exit_trend_active": (
                    int(bool(chart_decision.trend_active))
                    if chart_decision and chart_decision.trend_active is not None
                    else None
                ),
                "chart_exit_trade_days": (
                    chart_decision.trade_days if chart_decision else None
                ),
                "chart_exit_diagnostic": (
                    chart_decision.diagnostic if chart_decision else None
                ),
                "reason": reason,
            }
        )

    output = pd.DataFrame(results)
    if write_database:
        upsert_recommendations(engine, config, output)
    return output


def recommendation_preview_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the compact operator-facing view while retaining full DB detail."""
    if frame.empty or "chart_exit_action" not in frame.columns:
        return frame
    columns = [column for column in S3_OPERATIONAL_COLUMNS if column in frame.columns]
    return frame.loc[:, columns].copy()


def export_preview(frame: pd.DataFrame, output_path: str | Path | None) -> None:
    if output_path is None or frame.empty:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recommendation_preview_frame(frame).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )
