from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .features import FEATURE_COLUMNS


def quantile_key(value: float) -> str:
    return f"q{int(round(value * 100)):02d}"


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    error = y_true - y_pred
    return float(np.mean(np.maximum(alpha * error, (alpha - 1) * error)))


class ModelBundle:
    def __init__(self, directory: Path):
        self.directory = directory
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"模型元数据不存在：{metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._models: dict[str, Any] = {}

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata["feature_columns"])

    def _load_model(self, target: str, quantile: float) -> Any:
        key = f"{target}_{quantile_key(quantile)}"
        if key not in self._models:
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise RuntimeError("缺少 xgboost，请先安装 requirements.txt") from exc
            model = xgb.Booster()
            model.load_model(str(self.directory / f"{key}.json"))
            self._models[key] = model
        return self._models[key]

    def predict(
        self,
        frame: pd.DataFrame,
        target: str,
        quantile: float,
    ) -> np.ndarray:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("缺少 xgboost，请先安装 requirements.txt") from exc
        model = self._load_model(target, quantile)
        matrix = xgb.DMatrix(
            frame[self.feature_columns],
            feature_names=self.feature_columns,
        )
        return model.predict(matrix)


def _training_params(
    config: AppConfig,
    quantile: float,
    device: str,
) -> dict[str, Any]:
    return {
        "objective": "reg:quantileerror",
        "quantile_alpha": quantile,
        "max_depth": config.training.max_depth,
        "eta": config.training.learning_rate,
        "subsample": config.training.subsample,
        "colsample_bytree": config.training.colsample_bytree,
        "tree_method": "hist",
        "device": device,
        "nthread": config.performance.workers,
        "seed": config.training.random_seed,
    }


def _fit_with_gpu_fallback(
    config: AppConfig,
    quantile: float,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Any, str]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("缺少 xgboost，请先安装 requirements.txt") from exc

    training_matrix = xgb.QuantileDMatrix(
        x_train,
        label=y_train,
        feature_names=FEATURE_COLUMNS,
    )
    requested_device = "cuda" if config.training.use_gpu else "cpu"
    try:
        model = xgb.train(
            _training_params(config, quantile, requested_device),
            training_matrix,
            num_boost_round=config.training.n_estimators,
        )
        return model, requested_device
    except Exception:
        if requested_device == "cpu":
            raise
        model = xgb.train(
            _training_params(config, quantile, "cpu"),
            training_matrix,
            num_boost_round=config.training.n_estimators,
        )
        return model, "cpu-fallback"


def train_models(feature_frame: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    required = set(FEATURE_COLUMNS + ["trade_date", "next_high_return", "next_low_return"])
    missing = required.difference(feature_frame.columns)
    if missing:
        raise ValueError(f"训练数据缺少字段：{sorted(missing)}")

    training_columns = FEATURE_COLUMNS + [
        "trade_date",
        "next_high_return",
        "next_low_return",
    ]
    training_data = feature_frame[training_columns].dropna(
        subset=FEATURE_COLUMNS + ["next_high_return", "next_low_return"]
    )
    if training_data.empty:
        raise ValueError("清洗后没有可训练样本，请检查日线历史长度和字段质量")

    dates = np.sort(training_data["trade_date"].dt.normalize().unique())
    validation_days = config.training.validation_days
    if len(dates) <= validation_days + 20:
        raise ValueError(
            f"有效交易日仅 {len(dates)} 天，不足以预留 {validation_days} 天验证集"
        )
    split_date = pd.Timestamp(dates[-validation_days])
    train_mask = training_data["trade_date"] < split_date
    validation_mask = ~train_mask

    x_train = training_data.loc[train_mask, FEATURE_COLUMNS]
    x_validation = training_data.loc[validation_mask, FEATURE_COLUMNS]
    if x_train.empty or x_validation.empty:
        raise ValueError("时间切分后训练集或验证集为空")

    artifact_dir = config.artifacts_directory
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    devices: set[str] = set()

    targets = {
        "high": ("next_high_return", config.training.high_quantiles),
        "low": ("next_low_return", config.training.low_quantiles),
    }
    for target_name, (target_column, quantiles) in targets.items():
        y_train = training_data.loc[train_mask, target_column]
        y_validation = training_data.loc[validation_mask, target_column].to_numpy()
        for quantile in quantiles:
            key = f"{target_name}_{quantile_key(quantile)}"
            print(
                f"开始训练 {key}：训练 {len(x_train):,} 行，"
                f"验证 {len(x_validation):,} 行",
                flush=True,
            )
            model, device = _fit_with_gpu_fallback(
                config,
                quantile,
                x_train,
                y_train,
            )
            devices.add(device)
            model.save_model(str(artifact_dir / f"{key}.json"))
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise RuntimeError("缺少 xgboost，请先安装 requirements.txt") from exc
            validation_matrix = xgb.DMatrix(
                x_validation,
                feature_names=FEATURE_COLUMNS,
            )
            prediction = model.predict(validation_matrix)
            metrics[key] = {
                "pinball_loss": pinball_loss(y_validation, prediction, quantile),
                "quantile": quantile,
                "device": device,
            }
            print(
                f"完成 {key}：device={device}，"
                f"pinball_loss={metrics[key]['pinball_loss']:.8f}",
                flush=True,
            )

    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata = {
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "split_date": split_date.date().isoformat(),
        "devices": sorted(devices),
        "metrics": metrics,
        "training_config": asdict(config.training),
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata
