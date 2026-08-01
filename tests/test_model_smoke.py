import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tpsl.config import (
    AppConfig,
    BacktestConfig,
    ChartExitConfig,
    DataSyncConfig,
    DatabaseConfig,
    ExitMachineConfig,
    PerformanceConfig,
    PositionsConfig,
    RecommendationConfig,
    RiskFitConfig,
    StopTuningConfig,
    TrainingConfig,
    VolatilityStopConfig,
)
from tpsl.features import FEATURE_COLUMNS, build_feature_frame
from tpsl.model import ModelBundle, train_models
from tpsl.similarity import SimilarityIndex, build_similarity_index


class ModelSmokeTests(unittest.TestCase):
    def test_train_save_load_and_predict(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=190)
        rng = np.random.default_rng(42)
        rows = []
        for symbol_index in range(12):
            symbol = f"{symbol_index:06d}.SZ"
            industry = f"IND{symbol_index % 3}"
            close = 8.0 + symbol_index
            for day_index, trade_date in enumerate(dates):
                daily_return = (
                    0.0005
                    + 0.004 * np.sin(day_index / 11 + symbol_index)
                    + rng.normal(0, 0.008)
                )
                previous_close = close
                close = max(1.0, close * (1 + daily_return))
                open_price = previous_close * (1 + rng.normal(0, 0.003))
                high = max(open_price, close) * (1 + abs(rng.normal(0.006, 0.003)))
                low = min(open_price, close) * (1 - abs(rng.normal(0.006, 0.003)))
                volume = 1_000_000 * (1 + rng.normal(0, 0.12))
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": max(10_000, volume),
                        "amount": max(10_000, volume) * close,
                        "industry": industry,
                    }
                )

        feature_frame = build_feature_frame(
            pd.DataFrame(rows),
            workers=4,
            min_symbol_rows=100,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(
                database=DatabaseConfig(),
                positions=PositionsConfig(),
                performance=PerformanceConfig(workers=4),
                data_sync=DataSyncConfig(),
                training=TrainingConfig(
                    validation_days=30,
                    min_symbol_rows=100,
                    similarity_sample_limit=5_000,
                    n_estimators=8,
                    max_depth=3,
                    use_gpu=True,
                ),
                recommendation=RecommendationConfig(),
                backtest=BacktestConfig(),
                stop_tuning=StopTuningConfig(),
                risk_fit=RiskFitConfig(),
                volatility_stop=VolatilityStopConfig(),
                exit_machine=ExitMachineConfig(),
                chart_exit=ChartExitConfig(),
                artifacts_directory=Path(temp_dir),
                config_path=Path(temp_dir) / "config.toml",
            )
            metadata = train_models(feature_frame, config)
            self.assertGreater(metadata["training_rows"], 1_000)
            self.assertTrue(set(metadata["devices"]) & {"cuda", "cpu-fallback"})

            build_similarity_index(
                feature_frame,
                config.artifacts_directory,
                sample_limit=5_000,
                random_seed=42,
            )
            clean = feature_frame.dropna(subset=FEATURE_COLUMNS)
            current = clean.tail(2)
            bundle = ModelBundle(config.artifacts_directory)
            predictions = bundle.predict(current, "high", 0.50)
            self.assertEqual(len(predictions), 2)
            self.assertTrue(np.isfinite(predictions).all())

            similarity = SimilarityIndex(config.artifacts_directory)
            match = similarity.query(current.iloc[0], top_k=20)
            self.assertEqual(match["sample_count"], 20)
            self.assertTrue(np.isfinite(match["mean_distance"]))
            matches = similarity.query_batch(
                current,
                top_k=20,
                sample_limit=1000,
                batch_size=1,
            )
            self.assertEqual(len(matches), 2)
            self.assertTrue(np.isfinite(matches["mean_distance"]).all())


if __name__ == "__main__":
    unittest.main()
