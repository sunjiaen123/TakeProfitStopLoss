from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


class SimilarityIndex:
    def __init__(self, directory: Path):
        path = directory / "similarity_index.npz"
        if not path.exists():
            raise FileNotFoundError(f"相似形态索引不存在：{path}")
        data = np.load(path)
        self.matrix = data["matrix"]
        self.mean = data["mean"]
        self.scale = data["scale"]
        self.high_returns = data["high_returns"]
        self.low_returns = data["low_returns"]
        self.matrix_norm = np.sum(self.matrix * self.matrix, axis=1)

    def query(self, feature_row: pd.Series, top_k: int) -> dict[str, float | int]:
        vector = feature_row[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        standardized = (vector - self.mean) / self.scale
        distances = np.mean((self.matrix - standardized) ** 2, axis=1)
        sample_count = min(max(1, top_k), len(distances))
        indices = np.argpartition(distances, sample_count - 1)[:sample_count]
        high = self.high_returns[indices]
        low = self.low_returns[indices]
        return {
            "sample_count": int(sample_count),
            "median_high_return": float(np.nanmedian(high)),
            "low_return_q20": float(np.nanquantile(low, 0.20)),
            "mean_distance": float(np.mean(distances[indices])),
        }

    def query_batch(
        self,
        frame: pd.DataFrame,
        top_k: int,
        sample_limit: int | None = None,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "sample_count",
                    "median_high_return",
                    "low_return_q20",
                    "mean_distance",
                ],
                index=frame.index,
            )

        matrix = self.matrix
        matrix_norm = self.matrix_norm
        high_returns = self.high_returns
        low_returns = self.low_returns
        if sample_limit is not None and len(matrix) > sample_limit:
            sample_indices = np.linspace(
                0,
                len(matrix) - 1,
                num=sample_limit,
                dtype=np.int64,
            )
            matrix = matrix[sample_indices]
            matrix_norm = matrix_norm[sample_indices]
            high_returns = high_returns[sample_indices]
            low_returns = low_returns[sample_indices]

        sample_count = min(max(1, top_k), len(matrix))
        vectors = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        vectors = (vectors - self.mean) / self.scale
        results: list[dict[str, float | int]] = []

        for start in range(0, len(vectors), batch_size):
            batch = vectors[start : start + batch_size]
            batch_norm = np.sum(batch * batch, axis=1, keepdims=True)
            distances = (
                batch_norm
                + matrix_norm.reshape(1, -1)
                - 2.0 * (batch @ matrix.T)
            ) / matrix.shape[1]
            np.maximum(distances, 0.0, out=distances)
            nearest = np.argpartition(
                distances,
                sample_count - 1,
                axis=1,
            )[:, :sample_count]
            for row_index, indices in enumerate(nearest):
                high = high_returns[indices]
                low = low_returns[indices]
                results.append(
                    {
                        "sample_count": int(sample_count),
                        "median_high_return": float(np.nanmedian(high)),
                        "low_return_q20": float(np.nanquantile(low, 0.20)),
                        "mean_distance": float(
                            np.mean(distances[row_index, indices])
                        ),
                    }
                )
        return pd.DataFrame(results, index=frame.index)


def build_similarity_index(
    feature_frame: pd.DataFrame,
    directory: Path,
    sample_limit: int,
    random_seed: int,
) -> dict[str, int]:
    clean = feature_frame.dropna(
        subset=FEATURE_COLUMNS + ["next_high_return", "next_low_return"]
    ).copy()
    if clean.empty:
        raise ValueError("没有可用于构建相似形态索引的样本")
    if len(clean) > sample_limit:
        clean = clean.sample(n=sample_limit, random_state=random_seed)

    matrix = clean[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    standardized = ((matrix - mean) / scale).astype(np.float32)

    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "similarity_index.npz",
        matrix=standardized,
        mean=mean,
        scale=scale,
        high_returns=clean["next_high_return"].to_numpy(dtype=np.float32),
        low_returns=clean["next_low_return"].to_numpy(dtype=np.float32),
    )
    metadata = {
        "sample_count": int(len(clean)),
        "feature_count": len(FEATURE_COLUMNS),
    }
    (directory / "similarity_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata
