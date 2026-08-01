from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import _select_evaluation_symbols
from .config import AppConfig
from .db import load_bars
from .features import build_feature_frame
from .holding_backtest import _predict_stops
from .risk_fit import _promotion_diagnostics
from .stop_tuning import _build_paths, _entry_dates, _path_end_date, _simulate_path, summarize_paths


VOLATILITY_STOP_GAP_COLUMN = "volatility_stop_gap_pct"
VOLATILITY_SIGMA_STOCK_COLUMN = "volatility_sigma_stock"
VOLATILITY_SIGMA_POOL_COLUMN = "volatility_sigma_pool"
VOLATILITY_SIGMA_USED_COLUMN = "volatility_sigma_used"
VOLATILITY_HISTORY_DAYS_COLUMN = "volatility_history_days"
VOLATILITY_CREDIBILITY_COLUMN = "volatility_credibility"
VOLATILITY_METADATA_FILENAME = "metadata.json"


def volatility_stop_base_directory(config: AppConfig) -> Path:
    return config.artifacts_directory / "volatility-stop"


def _volatility_stop_directory(config: AppConfig) -> Path | None:
    base = volatility_stop_base_directory(config)
    pointer_path = base / "current.json"
    if not pointer_path.exists():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    relative_directory = str(pointer.get("directory", "")).strip()
    if not relative_directory:
        return None
    candidate = (base / relative_directory).resolve()
    base_resolved = base.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise RuntimeError(f"波动止损模型指针越界：{relative_directory}") from exc
    return candidate


def load_volatility_stop_state(config: AppConfig) -> dict[str, Any]:
    directory = _volatility_stop_directory(config)
    if directory is None:
        return {
            "version": "config",
            "source": "config",
            "k": float(config.volatility_stop.k),
        }
    metadata_path = directory / VOLATILITY_METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"波动止损元数据不存在：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "version": str(metadata["model_version"]),
        "source": str(directory),
        "k": float(metadata["selected_k"]),
        "metadata": metadata,
    }


def _clip_gap(values: pd.Series | np.ndarray, config: AppConfig) -> np.ndarray:
    return np.clip(
        np.asarray(values, dtype=float),
        float(config.volatility_stop.gap_min),
        float(config.volatility_stop.gap_max),
    )


def apply_volatility_stop_k(
    frame: pd.DataFrame,
    config: AppConfig,
    *,
    k: float,
) -> pd.DataFrame:
    result = frame.copy()
    if VOLATILITY_SIGMA_USED_COLUMN not in result.columns:
        result = add_volatility_stop_gaps(result, config, k=k)
        return result
    adjustment = float(
        np.clip(
            config.volatility_stop.adjustment_factor,
            config.volatility_stop.adjustment_min,
            config.volatility_stop.adjustment_max,
        )
    )
    result[VOLATILITY_STOP_GAP_COLUMN] = _clip_gap(
        float(k) * result[VOLATILITY_SIGMA_USED_COLUMN].astype(float) * adjustment,
        config,
    )
    return result


def add_volatility_stop_gaps(
    frame: pd.DataFrame,
    config: AppConfig,
    *,
    k: float | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required = {"symbol", "trade_date", "atr_14_pct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"波动止损缺少特征字段：{sorted(missing)}")

    result = frame.copy()
    if "industry" not in result.columns:
        result["industry"] = "UNKNOWN"
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result = result.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    atr = pd.to_numeric(result["atr_14_pct"], errors="coerce")
    atr = atr.where(np.isfinite(atr) & (atr > 0))
    result[VOLATILITY_SIGMA_STOCK_COLUMN] = atr

    market_pool = result.assign(_atr_for_pool=atr).groupby("trade_date")[
        "_atr_for_pool"
    ].transform("median")
    if config.volatility_stop.pool == "industry":
        industry_pool = result.assign(_atr_for_pool=atr).groupby(
            ["trade_date", "industry"]
        )["_atr_for_pool"].transform("median")
        pool_sigma = industry_pool.fillna(market_pool)
    else:
        pool_sigma = market_pool
    global_sigma = float(atr.median()) if pd.notna(atr.median()) else 0.02
    pool_sigma = pool_sigma.fillna(global_sigma).clip(lower=1e-6)
    stock_sigma = atr.fillna(pool_sigma).clip(lower=1e-6)

    lookback_days = int(config.volatility_stop.lookback_days)
    history_days = atr.notna().astype(float).groupby(result["symbol"]).transform(
        lambda values: values.rolling(lookback_days, min_periods=1).sum()
    )
    n0 = float(config.volatility_stop.shrink_n0)
    if n0 <= 0:
        credibility = pd.Series(1.0, index=result.index)
    else:
        credibility = history_days / (history_days + n0)
    sigma_used = credibility * stock_sigma + (1 - credibility) * pool_sigma

    result[VOLATILITY_SIGMA_STOCK_COLUMN] = stock_sigma.astype(float)
    result[VOLATILITY_SIGMA_POOL_COLUMN] = pool_sigma.astype(float)
    result[VOLATILITY_SIGMA_USED_COLUMN] = sigma_used.astype(float)
    result[VOLATILITY_HISTORY_DAYS_COLUMN] = history_days.astype(float)
    result[VOLATILITY_CREDIBILITY_COLUMN] = credibility.astype(float)
    return apply_volatility_stop_k(
        result,
        config,
        k=float(config.volatility_stop.k if k is None else k),
    )


def _prepare_volatility_features(
    engine: Any,
    config: AppConfig,
    earliest_entry: date,
    end_date: date,
    symbols: list[str],
) -> pd.DataFrame:
    feature_start = earliest_entry - timedelta(
        days=max(260, int(config.volatility_stop.lookback_days) + 60)
    )
    bars = load_bars(engine, config, feature_start, end_date)
    if bars.empty:
        raise RuntimeError("波动止损拟合范围没有行情数据")
    print(f"波动止损特征行情读取完成：{len(bars):,} 行", flush=True)
    features = build_feature_frame(
        bars,
        workers=config.performance.workers,
        min_symbol_rows=25,
    )
    features = add_volatility_stop_gaps(features, config)
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


def _simulate_volatility_candidates(
    paths: list[dict[str, Any]],
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    volatility_records: list[dict[str, Any]] = []
    fixed_records: list[dict[str, Any]] = []
    for path_id, path in enumerate(paths):
        for k in config.volatility_stop.k_values:
            path_with_gap = {
                **path,
                "rows": apply_volatility_stop_k(path["rows"], config, k=float(k)),
            }
            simulation = _simulate_path(
                path=path_with_gap,
                config=config,
                stop_order_type=config.volatility_stop.stop_order_type,
                atr_multiplier=0.0,
                limit_slippage_pct=config.volatility_stop.limit_slippage_pct,
                minimum_stop_gap_pct=float(config.recommendation.minimum_stop_gap_pct),
                minimum_stop_gap_column=VOLATILITY_STOP_GAP_COLUMN,
            )
            volatility_records.append(
                {
                    "path_id": path_id,
                    "k": float(k),
                    "strategy": "volatility",
                    **simulation,
                }
            )
        for fixed_gap in config.volatility_stop.fixed_baseline_gap_pcts:
            simulation = _simulate_path(
                path=path,
                config=config,
                stop_order_type=config.volatility_stop.stop_order_type,
                atr_multiplier=0.0,
                limit_slippage_pct=config.volatility_stop.limit_slippage_pct,
                minimum_stop_gap_pct=float(fixed_gap),
            )
            fixed_records.append(
                {
                    "path_id": path_id,
                    "fixed_gap_pct": float(fixed_gap),
                    "strategy": "fixed",
                    **simulation,
                }
            )
    return pd.DataFrame(volatility_records), pd.DataFrame(fixed_records)


def _select_by_constraint(
    candidates: pd.DataFrame,
    *,
    value_column: str,
    tolerance: float,
    max_loss_tie_tolerance: float = 0.0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for value, group in candidates.groupby(value_column, sort=True):
        summary = summarize_paths(group)
        shortfall = float(summary["average_hold_return"] - summary["average_strategy_return"])
        records.append(
            {
                value_column: float(value),
                **summary,
                "return_shortfall_vs_hold": shortfall,
                "return_constraint_tolerance": float(tolerance),
                "return_constraint_pass": bool(shortfall <= tolerance + 1e-12),
            }
        )
    if not records:
        raise RuntimeError("没有可用于选择参数的候选模拟结果")
    table = pd.DataFrame(records)
    eligible = table.loc[table["return_constraint_pass"]].copy()
    if eligible.empty:
        ranked = table.sort_values(
            [
                "return_shortfall_vs_hold",
                "average_strategy_max_loss",
                "whipsaw_rate",
                value_column,
            ],
            ascending=[True, False, True, False],
        ).reset_index(drop=True)
        rule = "no_candidate_met_return_constraint; chose smallest return shortfall"
    else:
        best_max_loss = float(eligible["average_strategy_max_loss"].max())
        near_best = eligible.loc[
            eligible["average_strategy_max_loss"]
            >= best_max_loss - float(max_loss_tie_tolerance)
        ].copy()
        ranked = near_best.sort_values(
            [
                "whipsaw_rate",
                value_column,
            ],
            ascending=[True, False],
        ).reset_index(drop=True)
        rule = "maximized average_strategy_max_loss under return constraint"
    selected = ranked.iloc[0].to_dict()
    return {
        "selected_value": float(selected[value_column]),
        "selection_rule": rule,
        "max_loss_tie_tolerance": float(max_loss_tie_tolerance),
        "selected": selected,
        "candidate_table": table.sort_values(value_column).to_dict("records"),
    }


def _shape_target_gap(config: AppConfig) -> float:
    configured = float(config.volatility_stop.shape_target_gap_pct)
    if configured > 0:
        return configured
    return float(config.recommendation.minimum_stop_gap_pct)


def _rows_for_path_ids(
    paths: list[dict[str, Any]],
    path_ids: set[int],
) -> pd.DataFrame:
    rows = [paths[int(path_id)]["rows"] for path_id in sorted(path_ids)]
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _mean_gap_for_k(
    sigma: np.ndarray,
    config: AppConfig,
    *,
    k: float,
) -> float:
    adjustment = float(
        np.clip(
            config.volatility_stop.adjustment_factor,
            config.volatility_stop.adjustment_min,
            config.volatility_stop.adjustment_max,
        )
    )
    gaps = np.clip(
        float(k) * sigma * adjustment,
        float(config.volatility_stop.gap_min),
        float(config.volatility_stop.gap_max),
    )
    return float(np.mean(gaps))


def _mean_stop_ratio_for_k(
    sigma: np.ndarray,
    config: AppConfig,
    *,
    k: float,
) -> float:
    adjustment = float(
        np.clip(
            config.volatility_stop.adjustment_factor,
            config.volatility_stop.adjustment_min,
            config.volatility_stop.adjustment_max,
        )
    )
    gaps = np.clip(
        float(k) * sigma * adjustment,
        float(config.volatility_stop.gap_min),
        float(config.volatility_stop.gap_max),
    )
    return float(np.mean(gaps / sigma))


def _valid_sigma_for_path_ids(
    paths: list[dict[str, Any]],
    path_ids: set[int],
) -> np.ndarray:
    rows = _rows_for_path_ids(paths, path_ids)
    if rows.empty or VOLATILITY_SIGMA_USED_COLUMN not in rows.columns:
        raise RuntimeError("cannot calibrate volatility stop: missing volatility sigma")
    sigma = (
        pd.to_numeric(rows[VOLATILITY_SIGMA_USED_COLUMN], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
    if sigma.size == 0:
        raise RuntimeError("cannot calibrate volatility stop: valid sigma is empty")
    return sigma


def _calibrate_k_to_target_gap(
    paths: list[dict[str, Any]],
    train_path_ids: set[int],
    config: AppConfig,
    *,
    target_gap: float,
) -> dict[str, Any]:
    rows = _rows_for_path_ids(paths, train_path_ids)
    if rows.empty or VOLATILITY_SIGMA_USED_COLUMN not in rows.columns:
        raise RuntimeError("无法按目标平均 gap 标定 k：缺少波动 sigma")
    sigma = (
        pd.to_numeric(rows[VOLATILITY_SIGMA_USED_COLUMN], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
    if sigma.size == 0:
        raise RuntimeError("无法按目标平均 gap 标定 k：有效 sigma 为空")

    target = float(
        np.clip(
            target_gap,
            float(config.volatility_stop.gap_min),
            float(config.volatility_stop.gap_max),
        )
    )
    low = 0.0
    high = max(float(max(config.volatility_stop.k_values)), float(config.volatility_stop.k), 1.0)
    while (
        _mean_gap_for_k(sigma, config, k=high) < target
        and high < 1_000
    ):
        high *= 2.0
    for _ in range(50):
        mid = (low + high) / 2.0
        if _mean_gap_for_k(sigma, config, k=mid) < target:
            low = mid
        else:
            high = mid
    selected_k = (low + high) / 2.0
    achieved = _mean_gap_for_k(sigma, config, k=selected_k)
    return {
        "selected_k": float(selected_k),
        "target_average_gap_pct": target,
        "achieved_train_average_gap_pct": achieved,
        "gap_error": float(achieved - target),
        "train_row_count": int(sigma.size),
        "selection_rule": (
            "shape-only calibration: choose k so mean(clamp(k * shrunk_ATR)) "
            "matches target average gap; no return/max-loss outcome used"
        ),
        "sigma_used_mean": float(np.mean(sigma)),
        "sigma_used_median": float(np.median(sigma)),
        "sigma_used_p10": float(np.quantile(sigma, 0.10)),
        "sigma_used_p90": float(np.quantile(sigma, 0.90)),
    }


def _calibrate_k_to_target_stop_ratio(
    paths: list[dict[str, Any]],
    train_path_ids: set[int],
    config: AppConfig,
    *,
    fixed_gap: float,
) -> dict[str, Any]:
    sigma = _valid_sigma_for_path_ids(paths, train_path_ids)
    fixed_gap = float(
        np.clip(
            fixed_gap,
            float(config.volatility_stop.gap_min),
            float(config.volatility_stop.gap_max),
        )
    )
    target_ratio = float(np.mean(fixed_gap / sigma))
    low = 0.0
    high = max(float(max(config.volatility_stop.k_values)), float(config.volatility_stop.k), 1.0)
    while (
        _mean_stop_ratio_for_k(sigma, config, k=high) < target_ratio
        and high < 1_000
    ):
        high *= 2.0
    for _ in range(50):
        mid = (low + high) / 2.0
        if _mean_stop_ratio_for_k(sigma, config, k=mid) < target_ratio:
            low = mid
        else:
            high = mid
    selected_k = (low + high) / 2.0
    achieved_ratio = _mean_stop_ratio_for_k(sigma, config, k=selected_k)
    achieved_gap = _mean_gap_for_k(sigma, config, k=selected_k)
    return {
        "selected_k": float(selected_k),
        "fixed_gap_pct": fixed_gap,
        "target_average_stop_ratio": target_ratio,
        "achieved_train_average_stop_ratio": achieved_ratio,
        "stop_ratio_error": float(achieved_ratio - target_ratio),
        "achieved_train_average_gap_pct": achieved_gap,
        "train_row_count": int(sigma.size),
        "selection_rule": (
            "shape-only calibration: choose k so mean(clamp(k * shrunk_ATR) "
            "/ shrunk_ATR) matches the fixed gap's mean stop multiple; "
            "no return/max-loss outcome used"
        ),
        "sigma_used_mean": float(np.mean(sigma)),
        "sigma_used_median": float(np.median(sigma)),
        "sigma_used_p10": float(np.quantile(sigma, 0.10)),
        "sigma_used_p90": float(np.quantile(sigma, 0.90)),
    }


def _entry_months(samples: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(samples["entry_date"]).dt.to_period("M").astype(str)


def _walk_forward_windows(samples: pd.DataFrame, config: AppConfig) -> list[dict[str, Any]]:
    months = sorted(_entry_months(samples).unique())
    train_months = int(config.volatility_stop.walk_forward_train_months)
    validation_months = int(config.volatility_stop.validation_months)
    step_months = int(config.volatility_stop.walk_forward_step_months)
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
        validation_values = months[
            validation_start : validation_start + validation_months
        ]
        windows.append(
            {
                "window_id": f"wf_{window_no:02d}",
                "train_months": train_values,
                "validation_months": validation_values,
                "train_path_ids": set(
                    samples.loc[entry_months.isin(train_values), "path_id"]
                    .astype(int)
                    .tolist()
                ),
                "validation_path_ids": set(
                    samples.loc[entry_months.isin(validation_values), "path_id"]
                    .astype(int)
                    .tolist()
                ),
            }
        )
        window_no += 1
    return windows


def _summarize_strategy_frame(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        strategy: summarize_paths(group)
        for strategy, group in frame.groupby("strategy", sort=True)
    }


def _diagnostics_against(
    summaries: dict[str, Any],
    benchmark: str,
) -> dict[str, Any]:
    diagnostics = _promotion_diagnostics(
        summaries["volatility"],
        summaries[benchmark],
    )
    diagnostics["benchmark"] = benchmark
    return diagnostics


def _combine_window_paths(
    *,
    window: dict[str, Any],
    paths: list[dict[str, Any]],
    config: AppConfig,
    selected_k: float,
    fixed_same_level_gap: float,
    matching_mode: str,
    fixed_best_gap: float | None = None,
) -> pd.DataFrame:
    validation_path_ids = window["validation_path_ids"]
    records: list[dict[str, Any]] = []
    for path_id in sorted(validation_path_ids):
        path = paths[int(path_id)]
        path_with_gap = {
            **path,
            "rows": apply_volatility_stop_k(path["rows"], config=config, k=selected_k),
        }
        volatility = _simulate_path(
            path=path_with_gap,
            config=config,
            stop_order_type=config.volatility_stop.stop_order_type,
            atr_multiplier=0.0,
            limit_slippage_pct=config.volatility_stop.limit_slippage_pct,
            minimum_stop_gap_pct=float(fixed_same_level_gap),
            minimum_stop_gap_column=VOLATILITY_STOP_GAP_COLUMN,
        )
        records.append(
            {
                "path_id": int(path_id),
                "matching_mode": matching_mode,
                "strategy": "volatility",
                "k": float(selected_k),
                "fixed_gap_pct": np.nan,
                **volatility,
            }
        )
        fixed_same_level = _simulate_path(
            path=path,
            config=config,
            stop_order_type=config.volatility_stop.stop_order_type,
            atr_multiplier=0.0,
            limit_slippage_pct=config.volatility_stop.limit_slippage_pct,
            minimum_stop_gap_pct=float(fixed_same_level_gap),
        )
        records.append(
            {
                "path_id": int(path_id),
                "matching_mode": matching_mode,
                "strategy": "fixed_same_level",
                "k": np.nan,
                "fixed_gap_pct": float(fixed_same_level_gap),
                **fixed_same_level,
            }
        )
        if fixed_best_gap is not None:
            fixed_best = _simulate_path(
                path=path,
                config=config,
                stop_order_type=config.volatility_stop.stop_order_type,
                atr_multiplier=0.0,
                limit_slippage_pct=config.volatility_stop.limit_slippage_pct,
                minimum_stop_gap_pct=float(fixed_best_gap),
            )
            records.append(
                {
                    "path_id": int(path_id),
                    "matching_mode": matching_mode,
                    "strategy": "fixed_train_best",
                    "k": np.nan,
                    "fixed_gap_pct": float(fixed_best_gap),
                    **fixed_best,
                }
            )
    combined = pd.DataFrame(records)
    combined["window_id"] = window["window_id"]
    return combined


def _bootstrap_deltas(
    frame: pd.DataFrame,
    *,
    benchmark: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0:
        return {"status": "skipped", "reason": "bootstrap disabled"}
    required = {"volatility", benchmark}
    if not required.issubset(set(frame["strategy"].astype(str))):
        return {"status": "skipped", "reason": "required strategies missing"}
    rng = np.random.default_rng(seed)
    path_ids = frame["path_id"].astype(int).drop_duplicates().to_numpy()
    if len(path_ids) < 2:
        return {"status": "skipped", "reason": "not enough paths"}
    index_by_path = {
        int(path_id): frame.index[frame["path_id"].astype(int) == int(path_id)].to_numpy()
        for path_id in path_ids
    }
    deltas: dict[str, list[float]] = {
        "average_strategy_return": [],
        "p05_strategy_return": [],
        "average_strategy_max_loss": [],
        "whipsaw_rate": [],
    }
    pass_count = 0
    for _ in range(iterations):
        sampled = rng.choice(path_ids, size=len(path_ids), replace=True)
        indices = np.concatenate([index_by_path[int(path_id)] for path_id in sampled])
        summaries = _summarize_strategy_frame(frame.loc[indices])
        diagnostics = _promotion_diagnostics(summaries["volatility"], summaries[benchmark])
        pass_count += int(diagnostics["pass"])
        for metric, value in diagnostics["deltas"].items():
            deltas[metric].append(float(value))
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
            for metric, values in deltas.items()
        },
    }


def _aggregate_diagnostics(records: list[dict[str, Any]], benchmark: str) -> dict[str, Any]:
    ok = [record for record in records if record.get("status") == "ok"]
    if not ok:
        return {"status": "skipped", "reason": "no successful windows"}
    deltas = pd.DataFrame(
        [record[f"diagnostics_vs_{benchmark}"]["deltas"] for record in ok]
    )
    delta_summary: dict[str, Any] = {}
    for metric in deltas.columns:
        values = deltas[metric].astype(float)
        se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        mean = float(values.mean())
        if metric == "whipsaw_rate":
            improves = bool(mean <= se)
        elif metric == "average_strategy_return":
            improves = bool(mean >= -se)
        else:
            improves = bool(mean > se)
        delta_summary[metric] = {
            "mean": mean,
            "standard_error_across_windows": se,
            "passes_directional_gate": improves,
        }
    pass_rate = float(
        np.mean([record[f"diagnostics_vs_{benchmark}"]["pass"] for record in ok])
    )
    criteria = {
        "pass_rate_at_least_half": pass_rate >= 0.5,
        "average_strategy_return_not_worse_than_one_se": delta_summary[
            "average_strategy_return"
        ]["passes_directional_gate"],
        "p05_strategy_return_improves_over_one_se": delta_summary[
            "p05_strategy_return"
        ]["passes_directional_gate"],
        "average_strategy_max_loss_improves_over_one_se": delta_summary[
            "average_strategy_max_loss"
        ]["passes_directional_gate"],
        "whipsaw_rate_not_higher_than_one_se": delta_summary["whipsaw_rate"][
            "passes_directional_gate"
        ],
    }
    return {
        "status": "ok",
        "benchmark": benchmark,
        "window_count": len(ok),
        "pass_rate": pass_rate,
        "criteria": criteria,
        "pass": bool(all(criteria.values())),
        "delta_summary": delta_summary,
    }


def _run_walk_forward(
    *,
    paths: list[dict[str, Any]],
    samples: pd.DataFrame,
    volatility_candidates: pd.DataFrame,
    fixed_candidates: pd.DataFrame,
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
    fixed_same_level_gap = _shape_target_gap(config)
    matching_modes = ("same_average_gap", "same_average_stop_ratio")
    records: list[dict[str, Any]] = []
    path_frames: list[pd.DataFrame] = []
    tolerance = float(config.volatility_stop.return_tolerance_pct)
    for window in windows:
        train_path_ids = window["train_path_ids"]
        train_fixed = fixed_candidates.loc[
            fixed_candidates["path_id"].astype(int).isin(train_path_ids)
        ]
        for matching_mode in matching_modes:
            try:
                if matching_mode == "same_average_gap":
                    k_selection = _calibrate_k_to_target_gap(
                        paths,
                        train_path_ids,
                        config,
                        target_gap=fixed_same_level_gap,
                    )
                else:
                    k_selection = _calibrate_k_to_target_stop_ratio(
                        paths,
                        train_path_ids,
                        config,
                        fixed_gap=fixed_same_level_gap,
                    )
                fixed_selection = _select_by_constraint(
                    train_fixed,
                    value_column="fixed_gap_pct",
                    tolerance=tolerance,
                    max_loss_tie_tolerance=float(
                        config.volatility_stop.max_loss_tie_tolerance_pct
                    ),
                )
                selected_k = float(k_selection["selected_k"])
                fixed_best_gap = float(fixed_selection["selected_value"])
                validation_paths = _combine_window_paths(
                    window=window,
                    paths=paths,
                    config=config,
                    selected_k=selected_k,
                    fixed_same_level_gap=float(fixed_same_level_gap),
                    matching_mode=matching_mode,
                    fixed_best_gap=fixed_best_gap,
                )
                summaries = _summarize_strategy_frame(validation_paths)
                diagnostics_same_level = _diagnostics_against(
                    summaries,
                    "fixed_same_level",
                )
                diagnostics_best = _diagnostics_against(summaries, "fixed_train_best")
                record = {
                    "window_id": window["window_id"],
                    "matching_mode": matching_mode,
                    "status": "ok",
                    "train_months": window["train_months"],
                    "validation_months": window["validation_months"],
                    "train_path_count": int(len(train_path_ids)),
                    "validation_path_count": int(len(window["validation_path_ids"])),
                    "selected_k": selected_k,
                    "shape_target_gap_pct": float(fixed_same_level_gap),
                    "fixed_same_level_gap_pct": float(fixed_same_level_gap),
                    "fixed_train_best_gap_pct": fixed_best_gap,
                    "k_selection": k_selection,
                    "fixed_selection": fixed_selection,
                    "validation_summaries": summaries,
                    "diagnostics_vs_fixed_same_level": diagnostics_same_level,
                    "diagnostics_vs_fixed_train_best": diagnostics_best,
                    "bootstrap_vs_fixed_same_level": _bootstrap_deltas(
                        validation_paths,
                        benchmark="fixed_same_level",
                        iterations=int(config.volatility_stop.bootstrap_samples),
                        seed=int(config.training.random_seed),
                    ),
                    "bootstrap_vs_fixed_train_best": _bootstrap_deltas(
                        validation_paths,
                        benchmark="fixed_train_best",
                        iterations=int(config.volatility_stop.bootstrap_samples),
                        seed=int(config.training.random_seed),
                    ),
                }
                records.append(record)
                path_frames.append(validation_paths)
            except Exception as exc:
                records.append(
                    {
                        "window_id": window["window_id"],
                        "matching_mode": matching_mode,
                        "status": "failed",
                        "train_months": window["train_months"],
                        "validation_months": window["validation_months"],
                        "error": str(exc),
                    }
                )
    paths = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
    mode_results: dict[str, Any] = {}
    for matching_mode in matching_modes:
        mode_records = [
            record for record in records if record.get("matching_mode") == matching_mode
        ]
        aggregate_same_level = _aggregate_diagnostics(mode_records, "fixed_same_level")
        aggregate_best = _aggregate_diagnostics(mode_records, "fixed_train_best")
        mode_results[matching_mode] = {
            "aggregate_vs_fixed_same_level": aggregate_same_level,
            "aggregate_vs_fixed_train_best": aggregate_best,
            "pass": bool(aggregate_same_level.get("pass", False)),
        }
    gap_result = mode_results["same_average_gap"]
    ratio_result = mode_results["same_average_stop_ratio"]
    shape_value_pass = any(bool(result["pass"]) for result in mode_results.values())
    promotion_pass = all(bool(result["pass"]) for result in mode_results.values())
    return (
        {
            "status": "ok" if path_frames else "failed",
            "records": records,
            "matching_modes": mode_results,
            "aggregate_vs_fixed_same_level": gap_result[
                "aggregate_vs_fixed_same_level"
            ],
            "aggregate_vs_fixed_train_best": gap_result[
                "aggregate_vs_fixed_train_best"
            ],
            "aggregate_vs_fixed_same_level_stop_ratio": ratio_result[
                "aggregate_vs_fixed_same_level"
            ],
            "shape_value_pass": bool(shape_value_pass),
            "promotion_pass": bool(promotion_pass),
            "pass": bool(promotion_pass),
            "pre_registered_decision_rule": (
                "freeze if both same_average_gap and same_average_stop_ratio fail; "
                "continue research only if at least one mode passes; promote only if both pass"
            ),
        },
        paths,
    )


def _promote_volatility_artifacts(source: Path, base: Path, version: str) -> Path:
    versions = base / "versions"
    target = versions / version
    temporary = versions / f".{version}.tmp"
    if target.exists():
        raise RuntimeError(f"波动止损版本已存在：{target}")
    if temporary.exists():
        shutil.rmtree(temporary)
    versions.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    os.replace(temporary, target)
    pointer = {
        "model_version": version,
        "directory": f"versions/{version}",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    pointer_tmp = base / "current.json.tmp"
    pointer_tmp.write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(pointer_tmp, base / "current.json")
    return target


def _read_pointer(base: Path) -> dict[str, Any] | None:
    pointer = base / "current.json"
    if not pointer.exists():
        return None
    try:
        return json.loads(pointer.read_text(encoding="utf-8"))
    except Exception:
        return {"unreadable_pointer": str(pointer)}


def _append_audit(base: Path, record: dict[str, Any]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    audit = base / "promotions.jsonl"
    with audit.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def fit_volatility_stop(
    engine: Any,
    config: AppConfig,
    *,
    start_date: date,
    end_date: date,
    max_symbols: int | None = None,
    output_directory: str | Path = "output/volatility-stop",
    save_model: bool = False,
    force_promote: bool = False,
) -> dict[str, Any]:
    if start_date > end_date:
        raise ValueError("波动止损拟合开始日期不能晚于结束日期")
    max_symbols = max_symbols or config.volatility_stop.max_symbols
    entry_dates = _entry_dates(engine, start_date, end_date)
    if not entry_dates:
        raise RuntimeError("波动止损拟合范围没有交易日")
    path_end_date = _path_end_date(
        engine,
        entry_dates[-1],
        max(config.volatility_stop.holding_days),
    )
    symbols = _select_evaluation_symbols(engine, config, start_date, max_symbols)
    if not symbols:
        raise RuntimeError("没有可用于波动止损拟合的股票")

    features = _prepare_volatility_features(
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
        config.volatility_stop.holding_days,
    )
    if not paths:
        raise RuntimeError("没有构建出完整的波动止损拟合路径")

    run_id = datetime.now(timezone.utc).strftime("vs_%Y%m%dT%H%M%SZ")
    output_root = Path(output_directory).resolve() / run_id
    model_output = output_root / "model"
    model_output.mkdir(parents=True, exist_ok=True)
    print(
        f"波动止损拟合：{len(symbols)} 支股票，{len(entry_dates)} 个建仓月，"
        f"{len(paths)} 条路径，{len(config.volatility_stop.k_values)} 个 k 候选",
        flush=True,
    )
    samples = _path_records(paths)
    volatility_candidates, fixed_candidates = _simulate_volatility_candidates(
        paths,
        config,
    )
    walk_forward, walk_forward_paths = _run_walk_forward(
        paths=paths,
        samples=samples,
        volatility_candidates=volatility_candidates,
        fixed_candidates=fixed_candidates,
        config=config,
    )
    selected_k_by_mode: dict[str, float] = {}
    for matching_mode in ("same_average_gap", "same_average_stop_ratio"):
        values = [
            float(record["selected_k"])
            for record in walk_forward.get("records", [])
            if record.get("status") == "ok"
            and record.get("matching_mode") == matching_mode
        ]
        if values:
            selected_k_by_mode[matching_mode] = float(np.median(values))
    selected_k = selected_k_by_mode.get(
        "same_average_gap",
        float(config.volatility_stop.k),
    )
    promotion_diagnostics = {
        "pass": bool(walk_forward.get("promotion_pass", False)),
        "shape_value_pass": bool(walk_forward.get("shape_value_pass", False)),
        "criteria": {
            "same_average_gap_pass": bool(
                walk_forward.get("matching_modes", {})
                .get("same_average_gap", {})
                .get("pass", False)
            ),
            "same_average_stop_ratio_pass": bool(
                walk_forward.get("matching_modes", {})
                .get("same_average_stop_ratio", {})
                .get("pass", False)
            ),
            "promotion_requires_both_modes": True,
        },
        "walk_forward": {
            "matching_modes": walk_forward.get("matching_modes"),
            "aggregate_vs_fixed_same_level": walk_forward.get(
                "aggregate_vs_fixed_same_level"
            ),
            "aggregate_vs_fixed_same_level_stop_ratio": walk_forward.get(
                "aggregate_vs_fixed_same_level_stop_ratio"
            ),
            "aggregate_vs_fixed_train_best": walk_forward.get(
                "aggregate_vs_fixed_train_best"
            ),
        },
    }

    samples_path = output_root / "paths.csv"
    volatility_candidates_path = output_root / "volatility_candidates.csv"
    fixed_candidates_path = output_root / "fixed_candidates.csv"
    walk_forward_paths_path = output_root / "walk_forward_paths.csv"
    walk_forward_summary_path = output_root / "walk_forward_summary.json"
    summary_path = output_root / "summary.json"
    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
    volatility_candidates.to_csv(
        volatility_candidates_path,
        index=False,
        encoding="utf-8-sig",
    )
    fixed_candidates.to_csv(
        fixed_candidates_path,
        index=False,
        encoding="utf-8-sig",
    )
    if not walk_forward_paths.empty:
        walk_forward_paths.to_csv(
            walk_forward_paths_path,
            index=False,
            encoding="utf-8-sig",
        )
    walk_forward_summary_path.write_text(
        json.dumps(walk_forward, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    metadata = {
        "model_version": run_id,
        "model_type": "volatility_layered_stop",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_k": selected_k,
        "selected_k_by_mode": selected_k_by_mode,
        "shape_target_gap_pct": _shape_target_gap(config),
        "selection_method": "median_same_average_gap_k_across_successful_walk_forward_windows",
        "volatility_stop_config": asdict(config.volatility_stop),
        "recommendation_config": asdict(config.recommendation),
        "promotion_diagnostics": promotion_diagnostics,
        "walk_forward_summary_path": str(walk_forward_summary_path),
    }
    (model_output / VOLATILITY_METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    artifact_directory: str | None = None
    promotion_action = "report_only"
    promotion_blocked = False
    base = volatility_stop_base_directory(config)
    if save_model:
        previous_pointer = _read_pointer(base)
        if not promotion_diagnostics["pass"] and not force_promote:
            promotion_blocked = True
            promotion_action = "blocked"
            _append_audit(
                base,
                {
                    "action": promotion_action,
                    "model_version": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "previous_pointer": previous_pointer,
                    "promotion_diagnostics": promotion_diagnostics,
                    "output_directory": str(output_root),
                    "reason": "promotion diagnostics failed; rerun with --force-promote to override",
                },
            )
        else:
            destination = _promote_volatility_artifacts(model_output, base, run_id)
            artifact_directory = str(destination)
            promotion_action = "force_promoted" if force_promote else "promoted"
            _append_audit(
                base,
                {
                    "action": promotion_action,
                    "model_version": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "previous_pointer": previous_pointer,
                    "new_pointer": _read_pointer(base),
                    "promotion_diagnostics": promotion_diagnostics,
                    "artifact_directory": artifact_directory,
                    "output_directory": str(output_root),
                },
            )
    else:
        print(
            "fit-volatility-stop 当前为报告模式，未覆盖 artifacts/volatility-stop；"
            "如需晋级，请显式加 --promote。",
            flush=True,
        )

    summary = {
        "run_id": run_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "path_end_date": path_end_date.isoformat(),
        "symbol_count": len(symbols),
        "entry_dates": [value.isoformat() for value in entry_dates],
        "path_count": len(paths),
        "sample_count": int(len(samples)),
        "selected_k": selected_k,
        "selected_k_by_mode": selected_k_by_mode,
        "shape_target_gap_pct": _shape_target_gap(config),
        "shape_value_pass": bool(walk_forward.get("shape_value_pass", False)),
        "promotion_pass": bool(walk_forward.get("promotion_pass", False)),
        "walk_forward": walk_forward,
        "promotion_diagnostics": promotion_diagnostics,
        "promotion_action": promotion_action,
        "promotion_blocked": promotion_blocked,
        "force_promote": force_promote,
        "artifact_directory": artifact_directory,
        "output_directory": str(output_root),
        "model_output_directory": str(model_output),
        "paths_path": str(samples_path),
        "volatility_candidates_path": str(volatility_candidates_path),
        "fixed_candidates_path": str(fixed_candidates_path),
        "walk_forward_summary_path": str(walk_forward_summary_path),
        "walk_forward_paths_path": (
            str(walk_forward_paths_path) if not walk_forward_paths.empty else None
        ),
        "method": (
            "base_gap = clamp(k × shrunk(ATR14%), gap_min, gap_max)。"
            "σ_stock 使用当前 ATR14%，σ_pool 使用同日同行业 ATR% 中位数并退化到全市场；"
            "w=n/(n+n0) 做可信度收缩。shape-only 实验同时检验 same_average_gap "
            "和 same_average_stop_ratio 两个预注册口径，只比较 k×σ 横截面差异化分配"
            "是否优于全员同一固定 gap；"
            "模拟时 atr_stop_multiplier=0，避免重复计算波动。"
        ),
        "limitations": [
            "只实现波动基准层，ML 微调系数 f 恒为 1",
            "日线只能近似止损成交顺序，未计佣金、印花税和真实滑点",
            "硬最大亏损线和棘轮逻辑保持原样",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if promotion_blocked:
        raise RuntimeError(
            "波动止损晋级检查未通过，已阻止覆盖生产参数。"
            f"报告已保存：{summary_path}"
        )
    return summary
