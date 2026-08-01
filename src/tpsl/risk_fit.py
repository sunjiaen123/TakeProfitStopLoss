from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import _select_evaluation_symbols
from .config import AppConfig
from .features import FEATURE_COLUMNS
from .holding_backtest import _predict_stops, _prepare_position_features
from .stop_tuning import _build_paths, _entry_dates, _path_end_date, _simulate_path, summarize_paths


RISK_MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "predicted_high_return",
    "predicted_low_return",
    "median_high_return",
    "low_return_q20",
    "mean_distance",
    "sample_count",
]

PROXY_GAP_FEATURE = "candidate_stop_gap_pct"
PROXY_FEATURE_COLUMNS = RISK_MODEL_FEATURE_COLUMNS + [PROXY_GAP_FEATURE]

RISK_MODEL_FILENAME = "stop_gap_model.json"
RISK_METADATA_FILENAME = "metadata.json"
RISK_PROXY_MODEL_TYPE = "risk_aligned_proxy"


def risk_model_directory(config: AppConfig) -> Path:
    base_directory = config.artifacts_directory / "risk"
    pointer_path = base_directory / "current.json"
    if not pointer_path.exists():
        return base_directory
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    relative_directory = str(pointer.get("directory", "")).strip()
    if not relative_directory:
        return base_directory
    candidate = (base_directory / relative_directory).resolve()
    base_resolved = base_directory.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise RuntimeError(
            f"动态风控模型指针越界：{relative_directory}"
        ) from exc
    return candidate


def build_risk_feature_frame(
    row: pd.Series | dict[str, Any],
    neighbors: dict[str, Any] | pd.Series | None = None,
) -> pd.DataFrame:
    source: dict[str, Any]
    if isinstance(row, pd.Series):
        source = row.to_dict()
    else:
        source = dict(row)
    if neighbors is not None:
        if isinstance(neighbors, pd.Series):
            source.update(neighbors.to_dict())
        else:
            source.update(dict(neighbors))
    record: dict[str, float] = {}
    for column in RISK_MODEL_FEATURE_COLUMNS:
        value = source.get(column, np.nan)
        record[column] = float(value) if pd.notna(value) else np.nan
    return pd.DataFrame([record], columns=RISK_MODEL_FEATURE_COLUMNS)


class RiskStopModel:
    def __init__(self, directory: Path):
        self.directory = directory
        metadata_path = directory / RISK_METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(f"动态风控模型元数据不存在：{metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._model: Any | None = None

    @classmethod
    def exists(cls, config: AppConfig) -> bool:
        try:
            directory = risk_model_directory(config)
        except Exception:
            return False
        return (
            (directory / RISK_METADATA_FILENAME).exists()
            and (directory / RISK_MODEL_FILENAME).exists()
        )

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata["feature_columns"])

    @property
    def model_type(self) -> str:
        return str(self.metadata.get("model_type", "legacy_gap_regression"))

    @property
    def base_feature_columns(self) -> list[str]:
        return list(self.metadata.get("base_feature_columns", self.feature_columns))

    @property
    def candidate_stop_gap_pcts(self) -> list[float]:
        values = self.metadata.get("candidate_stop_gap_pcts")
        if values is None:
            return []
        return [float(value) for value in values]

    @property
    def min_stop_gap_pct(self) -> float:
        return float(self.metadata.get("min_stop_gap_pct", 0.005))

    @property
    def max_stop_gap_pct(self) -> float:
        return float(self.metadata.get("max_stop_gap_pct", 0.100))

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise RuntimeError("缺少 xgboost，请先安装 requirements.txt") from exc
            model = xgb.Booster()
            model.load_model(str(self.directory / RISK_MODEL_FILENAME))
            self._model = model
        return self._model

    def predict_gap(self, frame: pd.DataFrame) -> np.ndarray:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("缺少 xgboost，请先安装 requirements.txt") from exc
        if self.model_type == RISK_PROXY_MODEL_TYPE:
            gaps = self.candidate_stop_gap_pcts
            if not gaps:
                raise ValueError("动态风控 proxy 模型缺少 candidate_stop_gap_pcts")
            missing = set(self.base_feature_columns).difference(frame.columns)
            if missing:
                raise ValueError(f"动态风控特征缺失：{sorted(missing)}")
            expanded_parts: list[pd.DataFrame] = []
            for gap in gaps:
                part = frame[self.base_feature_columns].copy()
                part[PROXY_GAP_FEATURE] = float(gap)
                expanded_parts.append(part)
            expanded = pd.concat(expanded_parts, ignore_index=True)
            matrix = xgb.DMatrix(
                expanded[self.feature_columns],
                feature_names=self.feature_columns,
            )
            scores = self._load_model().predict(matrix).reshape(len(gaps), len(frame)).T
            selected = np.asarray(gaps, dtype=float)[np.argmax(scores, axis=1)]
            return np.clip(
                selected,
                self.min_stop_gap_pct,
                self.max_stop_gap_pct,
            )
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"动态风控特征缺失：{sorted(missing)}")
        matrix = xgb.DMatrix(
            frame[self.feature_columns],
            feature_names=self.feature_columns,
        )
        prediction = self._load_model().predict(matrix)
        return np.clip(
            prediction.astype(float),
            self.min_stop_gap_pct,
            self.max_stop_gap_pct,
        )


def _candidate_score(record: dict[str, Any], max_loss_pct: float) -> float:
    strategy_return = float(record["strategy_return"])
    excess_return = float(record["excess_return"])
    strategy_max_loss = float(record["strategy_max_loss"])
    over_budget = max(0.0, -strategy_max_loss - max_loss_pct)
    tail_loss = max(0.0, -strategy_return - max_loss_pct)
    unfilled = 1.0 if int(record["stop_limit_unfilled_count"]) > 0 else 0.0
    whipsaw = float(record["whipsaw"])
    return (
        strategy_return
        + 0.5 * excess_return
        - 3.0 * over_budget
        - 1.5 * tail_loss
        - 0.03 * unfilled
        - 0.02 * whipsaw
    )


def _path_feature_record(path_id: int, path: dict[str, Any]) -> dict[str, Any]:
    row = path["rows"].iloc[0]
    record: dict[str, Any] = {
        "path_id": path_id,
        "entry_date": path["entry_date"],
        "symbol": path["symbol"],
        "holding_days": int(path["holding_days"]),
        "entry_price": float(path["entry_price"]),
    }
    for column in RISK_MODEL_FEATURE_COLUMNS:
        value = row.get(column, np.nan)
        record[column] = float(value) if pd.notna(value) else np.nan
    return record


def _build_training_samples(
    *,
    paths: list[dict[str, Any]],
    config: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []

    for path_id, path in enumerate(paths):
        base_record = _path_feature_record(path_id, path)
        samples.append(base_record)
        for stop_gap in config.risk_fit.candidate_stop_gap_pcts:
            simulation = _simulate_path(
                path=path,
                config=config,
                stop_order_type="limit",
                atr_multiplier=config.risk_fit.atr_multiplier,
                limit_slippage_pct=config.risk_fit.limit_slippage_pct,
                minimum_stop_gap_pct=float(stop_gap),
            )
            candidate = {
                "path_id": path_id,
                "candidate_stop_gap_pct": float(stop_gap),
                **simulation,
            }
            candidate_records.append(candidate)

    sample_frame = pd.DataFrame(samples)
    candidate_frame = pd.DataFrame(candidate_records)
    if sample_frame.empty:
        raise RuntimeError("动态止损拟合没有生成有效训练样本")
    sample_frame = sample_frame.dropna(subset=RISK_MODEL_FEATURE_COLUMNS)
    if sample_frame.empty:
        raise RuntimeError("动态止损拟合样本特征缺失，清洗后为空")
    valid_path_ids = set(sample_frame["path_id"].astype(int))
    candidate_frame = candidate_frame.loc[
        candidate_frame["path_id"].astype(int).isin(valid_path_ids)
    ].reset_index(drop=True)
    if candidate_frame.empty:
        raise RuntimeError("动态止损拟合候选模拟为空")
    return sample_frame.reset_index(drop=True), candidate_frame


def _split_samples(
    samples: pd.DataFrame,
    validation_months: int,
) -> tuple[pd.Series, pd.Series, str]:
    entry_months = (
        pd.to_datetime(samples["entry_date"]).dt.to_period("M").astype(str)
    )
    unique_months = sorted(entry_months.unique())
    if len(unique_months) <= validation_months:
        raise ValueError(
            f"可用建仓月份仅 {len(unique_months)} 个，不足以预留 "
            f"{validation_months} 个月验证集"
        )
    split_month = unique_months[-validation_months]
    validation_mask = entry_months >= split_month
    train_mask = ~validation_mask
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("动态止损时间切分后训练集或验证集为空")
    return train_mask, validation_mask, split_month


def _risk_training_params(config: AppConfig, device: str) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": config.risk_fit.max_depth,
        "eta": config.risk_fit.learning_rate,
        "subsample": config.training.subsample,
        "colsample_bytree": config.training.colsample_bytree,
        "tree_method": "hist",
        "device": device,
        "nthread": config.performance.workers,
        "seed": config.training.random_seed,
    }


def _fit_xgboost(
    *,
    config: AppConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    feature_columns: list[str],
) -> tuple[Any, str, dict[str, float]]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("缺少 xgboost，请先安装 requirements.txt") from exc

    train_matrix = xgb.DMatrix(
        x_train,
        label=y_train,
        feature_names=feature_columns,
    )
    validation_matrix = xgb.DMatrix(
        x_validation,
        label=y_validation,
        feature_names=feature_columns,
    )
    requested_device = "cuda" if config.training.use_gpu else "cpu"
    try:
        model = xgb.train(
            _risk_training_params(config, requested_device),
            train_matrix,
            num_boost_round=config.risk_fit.n_estimators,
            evals=[(validation_matrix, "validation")],
            verbose_eval=False,
        )
        device = requested_device
    except Exception:
        if requested_device == "cpu":
            raise
        model = xgb.train(
            _risk_training_params(config, "cpu"),
            train_matrix,
            num_boost_round=config.risk_fit.n_estimators,
            evals=[(validation_matrix, "validation")],
            verbose_eval=False,
        )
        device = "cpu-fallback"

    prediction = model.predict(validation_matrix).astype(float)
    truth = y_validation.to_numpy(dtype=float)
    metrics = {
        "mae": float(np.mean(np.abs(prediction - truth))),
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "prediction_mean": float(np.mean(prediction)),
        "target_mean": float(np.mean(truth)),
    }
    return model, device, metrics


def _simulate_validation_strategy(
    *,
    validation_samples: pd.DataFrame,
    paths: list[dict[str, Any]],
    config: AppConfig,
    dynamic_gaps: np.ndarray,
    fixed_gaps: dict[str, float],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sample, dynamic_gap in zip(
        validation_samples.itertuples(index=False),
        dynamic_gaps,
        strict=False,
    ):
        path_id = int(sample.path_id)
        path = paths[path_id]
        dynamic = _simulate_path(
            path=path,
            config=config,
            stop_order_type="limit",
            atr_multiplier=config.risk_fit.atr_multiplier,
            limit_slippage_pct=config.risk_fit.limit_slippage_pct,
            minimum_stop_gap_pct=float(dynamic_gap),
        )
        for strategy, gap in fixed_gaps.items():
            simulation = _simulate_path(
                path=path,
                config=config,
                stop_order_type="limit",
                atr_multiplier=config.risk_fit.atr_multiplier,
                limit_slippage_pct=config.risk_fit.limit_slippage_pct,
                minimum_stop_gap_pct=float(gap),
            )
            record = {
                "path_id": path_id,
                "strategy": strategy,
                "predicted_stop_gap_pct": float(gap),
                **simulation,
            }
            records.append(record)
        for strategy, simulation, gap in (
            ("dynamic", dynamic, float(dynamic_gap)),
        ):
            record = {
                "path_id": path_id,
                "strategy": strategy,
                "predicted_stop_gap_pct": gap,
                **simulation,
            }
            records.append(record)
    return pd.DataFrame(records)


def _select_best_fixed_gap(
    candidate_scores: pd.DataFrame,
    train_path_ids: set[int],
    config: AppConfig,
) -> dict[str, Any]:
    train_candidates = candidate_scores.loc[
        candidate_scores["path_id"].astype(int).isin(train_path_ids)
    ].copy()
    if train_candidates.empty:
        raise RuntimeError("训练集候选止损距离打分为空，无法选择固定 baseline")
    baseline_min_gap = float(config.risk_fit.fixed_baseline_min_gap_pct)
    if baseline_min_gap <= 0:
        baseline_min_gap = float(config.recommendation.minimum_stop_gap_pct)
    train_candidates = train_candidates.loc[
        train_candidates["candidate_stop_gap_pct"].astype(float)
        >= baseline_min_gap - 1e-12
    ].copy()
    if train_candidates.empty:
        raise RuntimeError("固定 baseline 最小 gap 过滤后没有候选止损距离")
    tolerance = float(config.risk_fit.fixed_baseline_return_tolerance_pct)
    records: list[dict[str, Any]] = []
    for gap, group in train_candidates.groupby("candidate_stop_gap_pct", sort=True):
        summary = summarize_paths(group)
        return_shortfall = float(
            summary["average_hold_return"] - summary["average_strategy_return"]
        )
        records.append(
            {
                "candidate_stop_gap_pct": float(gap),
                **summary,
                "return_shortfall_vs_hold": return_shortfall,
                "baseline_min_gap_pct": baseline_min_gap,
                "return_constraint_tolerance": tolerance,
                "return_constraint_pass": bool(return_shortfall <= tolerance + 1e-12),
            }
        )
    table = pd.DataFrame(records)
    eligible = table.loc[table["return_constraint_pass"]].copy()
    if eligible.empty:
        ranked = table.sort_values(
            [
                "return_shortfall_vs_hold",
                "average_strategy_max_loss",
                "p05_strategy_return",
                "whipsaw_rate",
            ],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
        selection_rule = (
            "no_gap_met_return_constraint; chose smallest return shortfall, "
            "then safer max loss"
        )
    else:
        ranked = eligible.sort_values(
            [
                "average_strategy_max_loss",
                "p05_strategy_return",
                "average_strategy_return",
                "whipsaw_rate",
            ],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        selection_rule = (
            "maximized average_strategy_max_loss under return shortfall constraint"
        )
    selected = ranked.iloc[0].to_dict()
    return {
        "g_star": float(selected["candidate_stop_gap_pct"]),
        "selection_rule": selection_rule,
        "baseline_min_gap_pct": baseline_min_gap,
        "return_tolerance_pct": tolerance,
        "selected": selected,
        "gap_scores": table.sort_values("candidate_stop_gap_pct").to_dict("records"),
    }


def _risk_aligned_score(
    frame: pd.DataFrame,
    *,
    lambda_loss: float,
    lambda_whip: float,
    lambda_unfilled: float,
    lambda_drawdown: float = 0.0,
) -> pd.Series:
    return (
        frame["strategy_return"].astype(float)
        + lambda_loss * frame["strategy_max_loss"].astype(float)
        - lambda_whip * frame["whipsaw"].astype(float)
        - lambda_unfilled
        * (frame["stop_limit_unfilled_count"].astype(float) > 0).astype(float)
        + lambda_drawdown * frame["strategy_max_drawdown"].astype(float)
    )


def _gap_objective_table(
    candidates: pd.DataFrame,
    *,
    lambda_loss: float,
    lambda_whip: float,
    lambda_unfilled: float,
    lambda_drawdown: float = 0.0,
) -> pd.DataFrame:
    frame = candidates.copy()
    frame["raw_score"] = _risk_aligned_score(
        frame,
        lambda_loss=lambda_loss,
        lambda_whip=lambda_whip,
        lambda_unfilled=lambda_unfilled,
        lambda_drawdown=lambda_drawdown,
    )
    grouped = frame.groupby("candidate_stop_gap_pct", as_index=False).agg(
        objective=("raw_score", "mean"),
        average_strategy_return=("strategy_return", "mean"),
        average_strategy_max_loss=("strategy_max_loss", "mean"),
        average_strategy_max_drawdown=("strategy_max_drawdown", "mean"),
        whipsaw_rate=("whipsaw", "mean"),
        stop_limit_unfilled_rate=(
            "stop_limit_unfilled_count",
            lambda values: (values.astype(float) > 0).mean(),
        ),
        path_count=("path_id", "count"),
    )
    return grouped.sort_values("objective", ascending=False).reset_index(drop=True)


def _calibrate_score_lambdas(
    candidate_scores: pd.DataFrame,
    train_path_ids: set[int],
    config: AppConfig,
) -> dict[str, Any]:
    train_candidates = candidate_scores.loc[
        candidate_scores["path_id"].astype(int).isin(train_path_ids)
    ].copy()
    if train_candidates.empty:
        raise RuntimeError("训练集候选模拟为空，无法标定 risk-aligned score")

    lambda_loss = float(config.risk_fit.proxy_lambda_loss)
    lambda_whip = float(config.risk_fit.proxy_lambda_whip)
    lambda_unfilled = float(config.risk_fit.proxy_lambda_unfilled)
    lambda_drawdown = float(config.risk_fit.proxy_lambda_drawdown)
    selected_table = _gap_objective_table(
        train_candidates,
        lambda_loss=lambda_loss,
        lambda_whip=lambda_whip,
        lambda_unfilled=lambda_unfilled,
        lambda_drawdown=lambda_drawdown,
    )
    proxy_objective_best = selected_table.iloc[0].to_dict()
    return {
        "lambda_loss": lambda_loss,
        "lambda_whip": lambda_whip,
        "lambda_unfilled": lambda_unfilled,
        "lambda_drawdown": lambda_drawdown,
        "selection_method": "configured_proxy_risk_aversion",
        "proxy_objective_best_gap": float(
            proxy_objective_best["candidate_stop_gap_pct"]
        ),
        "gap_objectives": selected_table.to_dict("records"),
    }


def _build_proxy_training_frame(
    samples: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    lambdas: dict[str, Any],
) -> pd.DataFrame:
    feature_columns = ["path_id"] + RISK_MODEL_FEATURE_COLUMNS
    frame = candidate_scores.merge(
        samples[feature_columns],
        on="path_id",
        how="inner",
        validate="many_to_one",
    ).copy()
    frame["raw_score"] = _risk_aligned_score(
        frame,
        lambda_loss=float(lambdas["lambda_loss"]),
        lambda_whip=float(lambdas["lambda_whip"]),
        lambda_unfilled=float(lambdas["lambda_unfilled"]),
        lambda_drawdown=float(lambdas["lambda_drawdown"]),
    )
    frame["target_score"] = frame["raw_score"] - frame.groupby("path_id")[
        "raw_score"
    ].transform("mean")
    frame[PROXY_GAP_FEATURE] = frame[PROXY_GAP_FEATURE].astype(float)
    return frame.dropna(subset=PROXY_FEATURE_COLUMNS + ["target_score"]).reset_index(
        drop=True
    )


def _predict_proxy_gaps(
    model: Any,
    frame: pd.DataFrame,
    *,
    candidate_gaps: list[float],
) -> pd.DataFrame:
    import xgboost as xgb

    records: list[dict[str, Any]] = []
    for gap in candidate_gaps:
        part = frame[["path_id"] + RISK_MODEL_FEATURE_COLUMNS].copy()
        part[PROXY_GAP_FEATURE] = float(gap)
        records.append(part)
    expanded = pd.concat(records, ignore_index=True)
    matrix = xgb.DMatrix(
        expanded[PROXY_FEATURE_COLUMNS],
        feature_names=PROXY_FEATURE_COLUMNS,
    )
    expanded["predicted_score"] = model.predict(matrix)
    chosen = expanded.loc[
        expanded.groupby("path_id")["predicted_score"].idxmax()
    ].copy()
    return chosen[["path_id", PROXY_GAP_FEATURE, "predicted_score"]].rename(
        columns={PROXY_GAP_FEATURE: "predicted_stop_gap_pct"}
    )


def _summarize_validation_paths(
    validation_paths: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    overall: dict[str, Any] = {}
    by_holding_days: dict[str, Any] = {}
    by_entry_month: dict[str, Any] = {}
    for strategy, group in validation_paths.groupby("strategy", sort=True):
        overall[strategy] = summarize_paths(group)
        holding_summaries: dict[str, Any] = {}
        for holding_days, holding_group in group.groupby("holding_days", sort=True):
            holding_summaries[str(int(holding_days))] = summarize_paths(holding_group)
        by_holding_days[strategy] = holding_summaries

        month_frame = group.copy()
        month_frame["entry_month"] = (
            pd.to_datetime(month_frame["entry_date"]).dt.to_period("M").astype(str)
        )
        month_summaries: dict[str, Any] = {}
        for entry_month, month_group in month_frame.groupby("entry_month", sort=True):
            month_summaries[str(entry_month)] = summarize_paths(month_group)
        by_entry_month[strategy] = month_summaries
    return overall, by_holding_days, by_entry_month


def _promotion_diagnostics(
    dynamic_summary: dict[str, Any],
    benchmark_summary: dict[str, Any],
) -> dict[str, Any]:
    criteria = {
        "average_strategy_max_loss_not_worse": (
            dynamic_summary["average_strategy_max_loss"]
            >= benchmark_summary["average_strategy_max_loss"]
        ),
        "p05_strategy_return_not_much_worse": (
            dynamic_summary["p05_strategy_return"]
            >= benchmark_summary["p05_strategy_return"] - 0.002
        ),
        "average_strategy_return_not_much_worse": (
            dynamic_summary["average_strategy_return"]
            >= benchmark_summary["average_strategy_return"] - 0.002
        ),
        "whipsaw_rate_not_much_higher": (
            dynamic_summary["whipsaw_rate"]
            <= benchmark_summary["whipsaw_rate"] + 0.02
        ),
    }
    return {
        "benchmark": "fixed_train_best",
        "pass": bool(all(criteria.values())),
        "criteria": criteria,
        "deltas": {
            "average_strategy_return": (
                dynamic_summary["average_strategy_return"]
                - benchmark_summary["average_strategy_return"]
            ),
            "p05_strategy_return": (
                dynamic_summary["p05_strategy_return"]
                - benchmark_summary["p05_strategy_return"]
            ),
            "average_strategy_max_loss": (
                dynamic_summary["average_strategy_max_loss"]
                - benchmark_summary["average_strategy_max_loss"]
            ),
            "whipsaw_rate": (
                dynamic_summary["whipsaw_rate"]
                - benchmark_summary["whipsaw_rate"]
            ),
        },
    }


def _walk_forward_promotion_diagnostics(walk_forward: dict[str, Any]) -> dict[str, Any]:
    if walk_forward.get("status") != "ok":
        return {
            "pass": False,
            "criteria": {
                "walk_forward_available": False,
            },
            "reason": walk_forward.get("reason", "walk-forward not available"),
        }
    aggregate = walk_forward.get("aggregate", {})
    delta_summary = aggregate.get("delta_summary", {})

    def metric(name: str) -> dict[str, float]:
        value = delta_summary.get(name, {})
        return {
            "mean": float(value.get("mean", 0.0)),
            "standard_error": float(
                value.get("standard_error_across_windows", 0.0)
            ),
        }

    return_delta = metric("average_strategy_return")
    p05_delta = metric("p05_strategy_return")
    max_loss_delta = metric("average_strategy_max_loss")
    whipsaw_delta = metric("whipsaw_rate")
    criteria = {
        "walk_forward_available": True,
        "pass_rate_at_least_half": float(aggregate.get("pass_rate", 0.0)) >= 0.5,
        "average_strategy_max_loss_improves_over_one_se": (
            max_loss_delta["mean"] > max_loss_delta["standard_error"]
        ),
        "p05_strategy_return_improves_over_one_se": (
            p05_delta["mean"] > p05_delta["standard_error"]
        ),
        "average_strategy_return_not_worse_than_one_se": (
            return_delta["mean"] >= -return_delta["standard_error"]
        ),
        "whipsaw_rate_not_higher_than_one_se": (
            whipsaw_delta["mean"] <= whipsaw_delta["standard_error"]
        ),
    }
    return {
        "pass": bool(all(criteria.values())),
        "criteria": criteria,
        "pass_rate": float(aggregate.get("pass_rate", 0.0)),
        "window_count": int(aggregate.get("window_count", 0)),
        "delta_summary": delta_summary,
    }


def _combine_promotion_diagnostics(
    single_window_diagnostics: dict[str, Any],
    walk_forward: dict[str, Any],
) -> dict[str, Any]:
    walk_forward_diagnostics = _walk_forward_promotion_diagnostics(walk_forward)
    criteria = {
        "single_window_pass": bool(single_window_diagnostics["pass"]),
        "walk_forward_pass": bool(walk_forward_diagnostics["pass"]),
    }
    return {
        "benchmark": single_window_diagnostics["benchmark"],
        "pass": bool(all(criteria.values())),
        "criteria": criteria,
        "deltas": single_window_diagnostics["deltas"],
        "single_window": single_window_diagnostics,
        "walk_forward": walk_forward_diagnostics,
    }


def _bootstrap_promotion_deltas(
    validation_paths: pd.DataFrame,
    *,
    benchmark_strategy: str = "fixed_train_best",
    iterations: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    if iterations <= 0 or validation_paths.empty or "path_id" not in validation_paths:
        return {"status": "skipped", "reason": "no bootstrap iterations or path_id"}
    required = {"dynamic", benchmark_strategy}
    if not required.issubset(set(validation_paths["strategy"].astype(str))):
        return {"status": "skipped", "reason": "required strategies missing"}

    rng = np.random.default_rng(seed)
    path_ids = validation_paths["path_id"].astype(int).drop_duplicates().to_numpy()
    if len(path_ids) < 2:
        return {"status": "skipped", "reason": "not enough paths"}
    index_by_path = {
        int(path_id): validation_paths.index[
            validation_paths["path_id"].astype(int) == int(path_id)
        ].to_numpy()
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
        sampled_path_ids = rng.choice(path_ids, size=len(path_ids), replace=True)
        sampled_indices = np.concatenate(
            [index_by_path[int(path_id)] for path_id in sampled_path_ids]
        )
        sample = validation_paths.loc[sampled_indices]
        summaries, _, _ = _summarize_validation_paths(sample)
        diagnostics = _promotion_diagnostics(
            summaries["dynamic"],
            summaries[benchmark_strategy],
        )
        pass_count += int(diagnostics["pass"])
        for metric, value in diagnostics["deltas"].items():
            deltas[metric].append(float(value))

    metric_summary = {}
    for metric, values in deltas.items():
        array = np.asarray(values, dtype=float)
        metric_summary[metric] = {
            "mean": float(array.mean()),
            "standard_error": float(array.std(ddof=1)),
            "p05": float(np.quantile(array, 0.05)),
            "p95": float(np.quantile(array, 0.95)),
        }
    return {
        "status": "ok",
        "iterations": iterations,
        "path_count": int(len(path_ids)),
        "pass_rate": float(pass_count / iterations),
        "deltas": metric_summary,
    }


def _entry_month_series(samples: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(samples["entry_date"]).dt.to_period("M").astype(str)


def _walk_forward_windows(
    samples: pd.DataFrame,
    config: AppConfig,
) -> list[dict[str, Any]]:
    entry_months = _entry_month_series(samples)
    months = sorted(entry_months.unique())
    train_months = int(config.risk_fit.walk_forward_train_months)
    validation_months = int(config.risk_fit.validation_months)
    step_months = int(config.risk_fit.walk_forward_step_months)
    if len(months) < train_months + validation_months:
        return []
    windows: list[dict[str, Any]] = []
    window_number = 1
    for validation_start in range(
        train_months,
        len(months) - validation_months + 1,
        step_months,
    ):
        train_values = months[validation_start - train_months : validation_start]
        validation_values = months[
            validation_start : validation_start + validation_months
        ]
        train_mask = entry_months.isin(train_values)
        validation_mask = entry_months.isin(validation_values)
        if not train_mask.any() or not validation_mask.any():
            continue
        windows.append(
            {
                "window_id": f"wf_{window_number:02d}",
                "train_months": train_values,
                "validation_months": validation_values,
                "train_mask": train_mask,
                "validation_mask": validation_mask,
            }
        )
        window_number += 1
    return windows


def _aggregate_walk_forward_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [record for record in records if record.get("status") == "ok"]
    if not ok_records:
        return {"status": "skipped", "reason": "no successful windows"}
    delta_rows = [record["promotion_diagnostics"]["deltas"] for record in ok_records]
    delta_frame = pd.DataFrame(delta_rows)
    delta_summary: dict[str, Any] = {}
    for metric in delta_frame.columns:
        values = delta_frame[metric].astype(float)
        standard_error = (
            float(values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
        mean = float(values.mean())
        if metric == "whipsaw_rate":
            improves_over_one_se = bool(mean < -standard_error)
        else:
            improves_over_one_se = bool(mean > standard_error)
        delta_summary[metric] = {
            "mean": mean,
            "standard_error_across_windows": standard_error,
            "improves_over_one_se": improves_over_one_se,
        }
    return {
        "status": "ok",
        "window_count": len(ok_records),
        "pass_rate": float(
            np.mean(
                [record["promotion_diagnostics"]["pass"] for record in ok_records]
            )
        ),
        "delta_summary": delta_summary,
    }


def _run_walk_forward_evaluations(
    *,
    samples: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    paths: list[dict[str, Any]],
    config: AppConfig,
    candidate_gaps: list[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    windows = _walk_forward_windows(samples, config)
    if not windows:
        return (
            {
                "status": "skipped",
                "reason": "not enough entry months for configured walk-forward",
                "train_months": config.risk_fit.walk_forward_train_months,
                "validation_months": config.risk_fit.validation_months,
            },
            pd.DataFrame(),
        )

    records: list[dict[str, Any]] = []
    path_frames: list[pd.DataFrame] = []
    for window in windows:
        train_mask = window["train_mask"]
        validation_mask = window["validation_mask"]
        train_path_ids = set(samples.loc[train_mask, "path_id"].astype(int).tolist())
        validation_samples = samples.loc[validation_mask].copy()
        validation_path_ids = set(validation_samples["path_id"].astype(int).tolist())
        try:
            score_calibration = _calibrate_score_lambdas(
                candidate_scores,
                train_path_ids,
                config,
            )
            fixed_baseline_selection = _select_best_fixed_gap(
                candidate_scores,
                train_path_ids,
                config,
            )
            proxy_frame = _build_proxy_training_frame(
                samples,
                candidate_scores,
                score_calibration,
            )
            proxy_train_mask = proxy_frame["path_id"].astype(int).isin(train_path_ids)
            proxy_validation_mask = proxy_frame["path_id"].astype(int).isin(
                validation_path_ids
            )
            x_train = proxy_frame.loc[proxy_train_mask, PROXY_FEATURE_COLUMNS]
            y_train = proxy_frame.loc[proxy_train_mask, "target_score"]
            x_validation = proxy_frame.loc[
                proxy_validation_mask,
                PROXY_FEATURE_COLUMNS,
            ]
            y_validation = proxy_frame.loc[proxy_validation_mask, "target_score"]
            if x_train.empty or x_validation.empty:
                raise RuntimeError("empty train or validation frame")

            model, device, model_metrics = _fit_xgboost(
                config=config,
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
                feature_columns=PROXY_FEATURE_COLUMNS,
            )
            train_proxy_gaps = _predict_proxy_gaps(
                model,
                samples.loc[train_mask].copy(),
                candidate_gaps=candidate_gaps,
            )
            validation_proxy_gaps = _predict_proxy_gaps(
                model,
                validation_samples,
                candidate_gaps=candidate_gaps,
            )
            validation_prediction = validation_samples["path_id"].map(
                validation_proxy_gaps.set_index("path_id")["predicted_stop_gap_pct"]
            ).to_numpy(dtype=float)
            if not np.isfinite(validation_prediction).all():
                raise RuntimeError("proxy did not predict all validation gaps")

            train_best_fixed_gap = float(fixed_baseline_selection["g_star"])
            train_dynamic_mean_gap = float(
                train_proxy_gaps["predicted_stop_gap_pct"].mean()
            )
            fixed_gaps = {
                "fixed_config": float(config.recommendation.minimum_stop_gap_pct),
                "fixed_10pct": 0.10,
                "fixed_dynamic_train_mean": train_dynamic_mean_gap,
                "fixed_train_best": train_best_fixed_gap,
            }
            validation_paths = _simulate_validation_strategy(
                validation_samples=validation_samples,
                paths=paths,
                config=config,
                dynamic_gaps=validation_prediction,
                fixed_gaps=fixed_gaps,
            )
            validation_paths["window_id"] = window["window_id"]
            summaries, by_holding_days, by_entry_month = _summarize_validation_paths(
                validation_paths
            )
            diagnostics = _promotion_diagnostics(
                summaries["dynamic"],
                summaries["fixed_train_best"],
            )
            bootstrap = _bootstrap_promotion_deltas(
                validation_paths,
                iterations=int(config.risk_fit.walk_forward_bootstrap_samples),
                seed=int(config.training.random_seed),
            )
            records.append(
                {
                    "window_id": window["window_id"],
                    "status": "ok",
                    "train_months": window["train_months"],
                    "validation_months": window["validation_months"],
                    "train_path_count": int(len(train_path_ids)),
                    "validation_path_count": int(len(validation_samples)),
                    "training_rows": int(len(x_train)),
                    "validation_rows": int(len(x_validation)),
                    "device": device,
                    "model_metrics": model_metrics,
                    "score_calibration": score_calibration,
                    "fixed_baseline_selection": fixed_baseline_selection,
                    "train_best_fixed_gap_pct": train_best_fixed_gap,
                    "train_dynamic_mean_gap_pct": train_dynamic_mean_gap,
                    "validation_summaries": summaries,
                    "validation_by_holding_days": by_holding_days,
                    "validation_by_entry_month": by_entry_month,
                    "promotion_diagnostics": diagnostics,
                    "bootstrap": bootstrap,
                }
            )
            path_frames.append(validation_paths)
        except Exception as exc:
            records.append(
                {
                    "window_id": window["window_id"],
                    "status": "failed",
                    "train_months": window["train_months"],
                    "validation_months": window["validation_months"],
                    "error": str(exc),
                }
            )

    path_frame = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
    return (
        {
            "status": "ok" if any(record.get("status") == "ok" for record in records) else "failed",
            "records": records,
            "aggregate": _aggregate_walk_forward_records(records),
        },
        path_frame,
    )


def _promote_model_artifacts(
    source: Path,
    base_directory: Path,
    version: str,
) -> Path:
    versions_directory = base_directory / "versions"
    target = versions_directory / version
    temporary = versions_directory / f".{version}.tmp"
    if target.exists():
        raise RuntimeError(f"动态风控模型版本已存在：{target}")
    if temporary.exists():
        shutil.rmtree(temporary)
    versions_directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    os.replace(temporary, target)

    pointer = {
        "model_version": version,
        "directory": f"versions/{version}",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    pointer_path = base_directory / "current.json"
    pointer_temporary = base_directory / "current.json.tmp"
    pointer_temporary.write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(pointer_temporary, pointer_path)
    return target


def _read_current_pointer(base_directory: Path) -> dict[str, Any] | None:
    pointer_path = base_directory / "current.json"
    if not pointer_path.exists():
        return None
    try:
        return json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception:
        return {"unreadable_pointer": str(pointer_path)}


def _append_promotion_audit(
    base_directory: Path,
    record: dict[str, Any],
) -> None:
    base_directory.mkdir(parents=True, exist_ok=True)
    audit_path = base_directory / "promotions.jsonl"
    line = json.dumps(record, ensure_ascii=False, default=str)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def fit_risk_model(
    engine: Any,
    config: AppConfig,
    *,
    start_date: date,
    end_date: date,
    max_symbols: int | None = None,
    output_directory: str | Path = "output/risk-fit",
    save_model: bool = False,
    force_promote: bool = False,
) -> dict[str, Any]:
    if start_date > end_date:
        raise ValueError("动态止损拟合开始日期不能晚于结束日期")
    max_symbols = max_symbols or config.risk_fit.max_symbols
    entry_dates = _entry_dates(engine, start_date, end_date)
    if not entry_dates:
        raise RuntimeError("动态止损拟合范围没有交易日")

    max_holding_days = max(config.risk_fit.holding_days)
    path_end_date = _path_end_date(engine, entry_dates[-1], max_holding_days)
    symbols = _select_evaluation_symbols(engine, config, start_date, max_symbols)
    if not symbols:
        raise RuntimeError("没有可用于动态止损拟合的股票")

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
        config.risk_fit.holding_days,
    )
    if not paths:
        raise RuntimeError("没有构建出完整的动态止损拟合路径")

    run_id = datetime.now(timezone.utc).strftime("rf_%Y%m%dT%H%M%SZ")
    output_root = Path(output_directory).resolve() / run_id
    model_output = output_root / "model"
    output_root.mkdir(parents=True, exist_ok=True)
    model_output.mkdir(parents=True, exist_ok=True)

    print(
        f"动态止损拟合：{len(symbols)} 支股票，{len(entry_dates)} 个建仓月，"
        f"{len(paths)} 条路径，{len(config.risk_fit.candidate_stop_gap_pcts)} 个候选距离",
        flush=True,
    )
    samples, candidate_scores = _build_training_samples(
        paths=paths,
        config=config,
    )
    train_mask, validation_mask, split_month = _split_samples(
        samples,
        config.risk_fit.validation_months,
    )
    train_path_ids = set(samples.loc[train_mask, "path_id"].astype(int).tolist())
    validation_samples = samples.loc[validation_mask].copy()
    validation_path_ids = set(validation_samples["path_id"].astype(int).tolist())
    score_calibration = _calibrate_score_lambdas(
        candidate_scores,
        train_path_ids,
        config,
    )
    fixed_baseline_selection = _select_best_fixed_gap(
        candidate_scores,
        train_path_ids,
        config,
    )
    proxy_frame = _build_proxy_training_frame(
        samples,
        candidate_scores,
        score_calibration,
    )
    proxy_train_mask = proxy_frame["path_id"].astype(int).isin(train_path_ids)
    proxy_validation_mask = proxy_frame["path_id"].astype(int).isin(
        validation_path_ids
    )
    x_train = proxy_frame.loc[proxy_train_mask, PROXY_FEATURE_COLUMNS]
    y_train = proxy_frame.loc[proxy_train_mask, "target_score"]
    x_validation = proxy_frame.loc[
        proxy_validation_mask,
        PROXY_FEATURE_COLUMNS,
    ]
    y_validation = proxy_frame.loc[proxy_validation_mask, "target_score"]
    if x_train.empty or x_validation.empty:
        raise RuntimeError("risk-aligned proxy 训练集或验证集为空")

    print(
        f"risk-aligned proxy 训练：训练 {len(x_train):,} 条，验证 {len(x_validation):,} 条，"
        f"验证起始月份 {split_month}，g*={float(fixed_baseline_selection['g_star']):.2%}，"
        f"lambda_loss={float(score_calibration['lambda_loss']):.2f}",
        flush=True,
    )
    model, device, model_metrics = _fit_xgboost(
        config=config,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        feature_columns=PROXY_FEATURE_COLUMNS,
    )
    candidate_gaps = [
        float(value) for value in config.risk_fit.candidate_stop_gap_pcts
    ]
    train_proxy_gaps = _predict_proxy_gaps(
        model,
        samples.loc[train_mask].copy(),
        candidate_gaps=candidate_gaps,
    )
    validation_proxy_gaps = _predict_proxy_gaps(
        model,
        validation_samples,
        candidate_gaps=candidate_gaps,
    )
    validation_prediction = validation_samples["path_id"].map(
        validation_proxy_gaps.set_index("path_id")["predicted_stop_gap_pct"]
    ).to_numpy(dtype=float)
    if not np.isfinite(validation_prediction).all():
        raise RuntimeError("risk-aligned proxy 未能为全部验证路径生成 gap")
    train_best_fixed_gap = float(fixed_baseline_selection["g_star"])
    train_fixed_gap_scores = list(fixed_baseline_selection["gap_scores"])
    train_dynamic_mean_gap = float(
        train_proxy_gaps["predicted_stop_gap_pct"].mean()
    )
    fixed_gaps = {
        "fixed_config": float(config.recommendation.minimum_stop_gap_pct),
        "fixed_10pct": 0.10,
        "fixed_dynamic_train_mean": train_dynamic_mean_gap,
        "fixed_train_best": train_best_fixed_gap,
    }
    validation_paths = _simulate_validation_strategy(
        validation_samples=validation_samples,
        paths=paths,
        config=config,
        dynamic_gaps=validation_prediction,
        fixed_gaps=fixed_gaps,
    )
    validation_summaries, validation_by_holding_days, validation_by_entry_month = (
        _summarize_validation_paths(validation_paths)
    )
    fixed_summary = validation_summaries["fixed_config"]
    dynamic_summary = validation_summaries["dynamic"]
    single_window_promotion_diagnostics = _promotion_diagnostics(
        dynamic_summary,
        validation_summaries["fixed_train_best"],
    )
    print("walk-forward risk proxy 评测开始（report-only，不晋级）", flush=True)
    walk_forward, walk_forward_paths = _run_walk_forward_evaluations(
        samples=samples,
        candidate_scores=candidate_scores,
        paths=paths,
        config=config,
        candidate_gaps=candidate_gaps,
    )
    if walk_forward.get("status") == "skipped":
        print(f"walk-forward 已跳过：{walk_forward.get('reason')}", flush=True)
    else:
        aggregate = walk_forward.get("aggregate", {})
        print(
            "walk-forward 完成："
            f"{aggregate.get('window_count', 0)} 个成功窗口，"
            f"pass_rate={float(aggregate.get('pass_rate', 0.0)):.2%}",
            flush=True,
        )
    promotion_diagnostics = _combine_promotion_diagnostics(
        single_window_promotion_diagnostics,
        walk_forward,
    )

    model_version = run_id
    metadata = {
        "model_version": model_version,
        "model_type": RISK_PROXY_MODEL_TYPE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": PROXY_FEATURE_COLUMNS,
        "base_feature_columns": RISK_MODEL_FEATURE_COLUMNS,
        "candidate_stop_gap_pcts": candidate_gaps,
        "target": "risk_aligned_centered_score",
        "min_stop_gap_pct": config.risk_fit.min_stop_gap_pct,
        "max_stop_gap_pct": config.risk_fit.max_stop_gap_pct,
        "training_rows": int(len(x_train)),
        "validation_rows": int(len(x_validation)),
        "split_month": split_month,
        "device": device,
        "metrics": model_metrics,
        "score_calibration": score_calibration,
        "fixed_baseline_selection": fixed_baseline_selection,
        "fixed_validation_summary": fixed_summary,
        "dynamic_validation_summary": dynamic_summary,
        "validation_summaries": validation_summaries,
        "validation_by_holding_days": validation_by_holding_days,
        "validation_by_entry_month": validation_by_entry_month,
        "train_best_fixed_gap_pct": train_best_fixed_gap,
        "train_dynamic_mean_gap_pct": train_dynamic_mean_gap,
        "train_fixed_gap_scores": train_fixed_gap_scores,
        "promotion_diagnostics": promotion_diagnostics,
        "walk_forward": walk_forward,
        "risk_fit_config": asdict(config.risk_fit),
        "recommendation_config": asdict(config.recommendation),
    }
    model.save_model(str(model_output / RISK_MODEL_FILENAME))
    (model_output / RISK_METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    samples_path = output_root / "training_samples.csv"
    candidates_path = output_root / "candidate_scores.csv"
    proxy_path = output_root / "proxy_training_rows.csv"
    validation_path = output_root / "validation_paths.csv"
    walk_forward_path = output_root / "walk_forward_paths.csv"
    walk_forward_summary_path = output_root / "walk_forward_summary.json"
    summary_path = output_root / "summary.json"
    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
    candidate_scores.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    proxy_frame.to_csv(proxy_path, index=False, encoding="utf-8-sig")
    validation_paths.to_csv(validation_path, index=False, encoding="utf-8-sig")
    if not walk_forward_paths.empty:
        walk_forward_paths.to_csv(
            walk_forward_path,
            index=False,
            encoding="utf-8-sig",
        )
    walk_forward_summary_path.write_text(
        json.dumps(walk_forward, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    artifact_directory: str | None = None
    promotion_blocked = False
    promotion_action = "report_only"
    risk_artifact_base = config.artifacts_directory / "risk"
    if save_model:
        status = "PASS" if promotion_diagnostics["pass"] else "FAIL"
        previous_pointer = _read_current_pointer(risk_artifact_base)
        if not promotion_diagnostics["pass"] and not force_promote:
            promotion_blocked = True
            promotion_action = "blocked"
            _append_promotion_audit(
                risk_artifact_base,
                {
                    "action": promotion_action,
                    "model_version": model_version,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "previous_pointer": previous_pointer,
                    "promotion_diagnostics": promotion_diagnostics,
                    "output_directory": str(output_root),
                    "reason": "promotion diagnostics failed; rerun with --force-promote to override",
                },
            )
            print(
                f"动态风控模型晋级检查：{status}，对照=fixed_train_best，"
                f"train_best_gap={train_best_fixed_gap:.2%}。"
                "已拒绝晋级；如需强制覆盖，请显式加 --force-promote。",
                flush=True,
            )
        else:
            destination = _promote_model_artifacts(
                model_output,
                risk_artifact_base,
                model_version,
            )
            artifact_directory = str(destination)
            promotion_action = "force_promoted" if force_promote else "promoted"
            _append_promotion_audit(
                risk_artifact_base,
                {
                    "action": promotion_action,
                    "model_version": model_version,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "previous_pointer": previous_pointer,
                    "new_pointer": _read_current_pointer(risk_artifact_base),
                    "artifact_directory": artifact_directory,
                    "promotion_diagnostics": promotion_diagnostics,
                    "output_directory": str(output_root),
                },
            )
            print(
                f"动态风控模型晋级检查：{status}，对照=fixed_train_best，"
                f"train_best_gap={train_best_fixed_gap:.2%}。"
                f"已执行 {promotion_action}。",
                flush=True,
            )
    else:
        print(
            "fit-risk 当前为报告模式，未覆盖 artifacts/risk；"
            "如需晋级为生产风控模型，请显式加 --promote。",
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
        "training_rows": int(len(x_train)),
        "validation_rows": int(len(x_validation)),
        "split_month": split_month,
        "device": device,
        "model_type": RISK_PROXY_MODEL_TYPE,
        "model_metrics": model_metrics,
        "score_calibration": score_calibration,
        "fixed_baseline_selection": fixed_baseline_selection,
        "fixed_validation_summary": fixed_summary,
        "dynamic_validation_summary": dynamic_summary,
        "validation_summaries": validation_summaries,
        "validation_by_holding_days": validation_by_holding_days,
        "validation_by_entry_month": validation_by_entry_month,
        "train_best_fixed_gap_pct": train_best_fixed_gap,
        "train_dynamic_mean_gap_pct": train_dynamic_mean_gap,
        "train_fixed_gap_scores": train_fixed_gap_scores,
        "promotion_diagnostics": promotion_diagnostics,
        "walk_forward": walk_forward,
        "promotion_action": promotion_action,
        "promotion_blocked": promotion_blocked,
        "force_promote": force_promote,
        "model_output_directory": str(model_output),
        "artifact_directory": artifact_directory,
        "output_directory": str(output_root),
        "proxy_training_rows_path": str(proxy_path),
        "walk_forward_summary_path": str(walk_forward_summary_path),
        "walk_forward_paths_path": (
            str(walk_forward_path) if not walk_forward_paths.empty else None
        ),
        "method": (
            "对每条历史路径和候选止损距离生成 risk-aligned raw_score，"
            "再做 path 内均值中心化，训练 XGBoost proxy: "
            "centered_score = f(features, candidate_stop_gap_pct)。"
            "验证时对 gap 网格取预测 score 最大者，再按同一模拟器闭环评估。"
        ),
        "limitations": [
            "标签来自历史路径模拟，不是未来确定事实",
            "日线只能近似止损成交顺序，未计佣金、印花税和真实滑点",
            "硬最大亏损线仍由持仓 max_loss_pct 或 positions.default_max_loss_pct 控制",
            "股票池来自当前数据库可用股票，仍可能有幸存者偏差",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if promotion_blocked:
        raise RuntimeError(
            "动态风控模型晋级检查未通过，已阻止覆盖生产模型。"
            f"报告已保存：{summary_path}"
        )
    return summary
