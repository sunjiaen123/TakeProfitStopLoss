import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tpsl.features import FEATURE_COLUMNS
from tpsl.risk_fit import (
    RISK_MODEL_FEATURE_COLUMNS,
    _build_proxy_training_frame,
    _combine_promotion_diagnostics,
    _promote_model_artifacts,
    _promotion_diagnostics,
    _risk_aligned_score,
    _select_best_fixed_gap,
    _candidate_score,
    build_risk_feature_frame,
    risk_model_directory,
)


class RiskFitTests(unittest.TestCase):
    def test_candidate_score_penalizes_loss_budget_breach(self) -> None:
        safe = {
            "strategy_return": -0.01,
            "excess_return": 0.02,
            "strategy_max_loss": -0.03,
            "stop_limit_unfilled_count": 0,
            "whipsaw": 0,
        }
        breached = {
            "strategy_return": -0.01,
            "excess_return": 0.02,
            "strategy_max_loss": -0.09,
            "stop_limit_unfilled_count": 0,
            "whipsaw": 0,
        }
        self.assertGreater(
            _candidate_score(safe, 0.05),
            _candidate_score(breached, 0.05),
        )

    def test_build_risk_feature_frame_merges_signal_and_neighbors(self) -> None:
        signal = {column: 0.01 for column in FEATURE_COLUMNS}
        signal.update(
            {
                "predicted_high_return": 0.03,
                "predicted_low_return": -0.02,
            }
        )
        neighbors = {
            "median_high_return": 0.025,
            "low_return_q20": -0.018,
            "mean_distance": 1.2,
            "sample_count": 100,
        }
        frame = build_risk_feature_frame(pd.Series(signal), neighbors)
        self.assertEqual(list(frame.columns), RISK_MODEL_FEATURE_COLUMNS)
        self.assertEqual(len(frame), 1)
        self.assertTrue(np.isfinite(frame.to_numpy(dtype=float)).all())
        self.assertEqual(float(frame.loc[0, "sample_count"]), 100.0)

    def test_select_best_fixed_gap_uses_explicit_return_constraint(self) -> None:
        def row(path_id: int, gap: float, strategy_return: float, max_loss: float):
            hold_return = 0.010
            return {
                "path_id": path_id,
                "candidate_stop_gap_pct": gap,
                "stopped": 1,
                "stop_limit_unfilled_count": 0,
                "strategy_return": strategy_return,
                "hold_return": hold_return,
                "excess_return": strategy_return - hold_return,
                "strategy_max_drawdown": max_loss,
                "hold_max_drawdown": -0.050,
                "strategy_max_loss": max_loss,
                "hold_max_loss": -0.050,
                "loss_avoided": 0,
                "whipsaw": 0,
            }

        candidate_scores = pd.DataFrame(
            [
                row(1, 0.01, 0.008, -0.010),
                row(1, 0.05, 0.012, -0.040),
                row(2, 0.01, 0.008, -0.010),
                row(2, 0.05, 0.012, -0.040),
                row(9, 0.01, -0.100, -0.090),
                row(9, 0.05, 0.100, -0.001),
            ]
        )
        config = SimpleNamespace(
            risk_fit=SimpleNamespace(
                fixed_baseline_min_gap_pct=0.0,
                fixed_baseline_return_tolerance_pct=0.002,
            ),
            recommendation=SimpleNamespace(minimum_stop_gap_pct=0.0),
        )
        selection = _select_best_fixed_gap(candidate_scores, {1, 2}, config)
        self.assertEqual(selection["g_star"], 0.01)
        self.assertTrue(selection["selected"]["return_constraint_pass"])

    def test_select_best_fixed_gap_respects_production_floor(self) -> None:
        def row(gap: float, max_loss: float):
            return {
                "path_id": 1,
                "candidate_stop_gap_pct": gap,
                "stopped": 1,
                "stop_limit_unfilled_count": 0,
                "strategy_return": 0.02,
                "hold_return": -0.01,
                "excess_return": 0.03,
                "strategy_max_drawdown": max_loss,
                "hold_max_drawdown": -0.05,
                "strategy_max_loss": max_loss,
                "hold_max_loss": -0.05,
                "loss_avoided": 1,
                "whipsaw": 0,
            }

        candidate_scores = pd.DataFrame([row(0.005, -0.005), row(0.020, -0.020)])
        config = SimpleNamespace(
            risk_fit=SimpleNamespace(
                fixed_baseline_min_gap_pct=0.0,
                fixed_baseline_return_tolerance_pct=0.002,
            ),
            recommendation=SimpleNamespace(minimum_stop_gap_pct=0.020),
        )
        selection = _select_best_fixed_gap(candidate_scores, {1}, config)
        self.assertEqual(selection["g_star"], 0.020)

    def test_risk_aligned_score_uses_full_max_loss_linear_term(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "strategy_return": 0.01,
                    "strategy_max_loss": -0.02,
                    "strategy_max_drawdown": -0.03,
                    "whipsaw": 0,
                    "stop_limit_unfilled_count": 0,
                    "excess_return": 99.0,
                },
                {
                    "strategy_return": 0.01,
                    "strategy_max_loss": -0.04,
                    "strategy_max_drawdown": -0.03,
                    "whipsaw": 0,
                    "stop_limit_unfilled_count": 0,
                    "excess_return": -99.0,
                },
            ]
        )
        score = _risk_aligned_score(
            frame,
            lambda_loss=2.0,
            lambda_whip=0.0,
            lambda_unfilled=0.0,
        )
        self.assertAlmostEqual(float(score.iloc[0] - score.iloc[1]), 0.04)

    def test_proxy_training_target_is_centered_within_path(self) -> None:
        samples = pd.DataFrame(
            [
                {
                    "path_id": 1,
                    **{column: 0.01 for column in RISK_MODEL_FEATURE_COLUMNS},
                },
                {
                    "path_id": 2,
                    **{column: 0.02 for column in RISK_MODEL_FEATURE_COLUMNS},
                },
            ]
        )
        candidates = pd.DataFrame(
            [
                {
                    "path_id": path_id,
                    "candidate_stop_gap_pct": gap,
                    "strategy_return": 0.01 + gap,
                    "strategy_max_loss": -gap,
                    "strategy_max_drawdown": -gap,
                    "whipsaw": 0,
                    "stop_limit_unfilled_count": 0,
                }
                for path_id in (1, 2)
                for gap in (0.01, 0.02, 0.03)
            ]
        )
        frame = _build_proxy_training_frame(
            samples,
            candidates,
            {
                "lambda_loss": 1.0,
                "lambda_whip": 0.0,
                "lambda_unfilled": 0.0,
                "lambda_drawdown": 0.0,
            },
        )
        centered = frame.groupby("path_id")["target_score"].mean().abs()
        self.assertTrue((centered < 1e-12).all())

    def test_promote_model_artifacts_uses_current_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            (source / "metadata.json").write_text("{}", encoding="utf-8")
            (source / "stop_gap_model.json").write_text("model", encoding="utf-8")

            promoted = _promote_model_artifacts(source, root / "risk", "rf_test")
            self.assertTrue((promoted / "metadata.json").exists())
            self.assertTrue((root / "risk" / "current.json").exists())

            config = SimpleNamespace(artifacts_directory=root)
            self.assertEqual(risk_model_directory(config), promoted.resolve())

    def test_promotion_diagnostics_fails_when_max_loss_is_worse(self) -> None:
        dynamic = {
            "average_strategy_max_loss": -0.024,
            "p05_strategy_return": -0.053,
            "average_strategy_return": -0.004,
            "whipsaw_rate": 0.33,
        }
        benchmark = {
            "average_strategy_max_loss": -0.021,
            "p05_strategy_return": -0.053,
            "average_strategy_return": -0.007,
            "whipsaw_rate": 0.34,
        }
        diagnostics = _promotion_diagnostics(dynamic, benchmark)
        self.assertFalse(diagnostics["pass"])
        self.assertFalse(
            diagnostics["criteria"]["average_strategy_max_loss_not_worse"]
        )

    def test_combined_promotion_requires_walk_forward_pass(self) -> None:
        single = {
            "benchmark": "fixed_train_best",
            "pass": True,
            "criteria": {},
            "deltas": {
                "average_strategy_return": 0.0,
                "p05_strategy_return": 0.01,
                "average_strategy_max_loss": 0.01,
                "whipsaw_rate": 0.0,
            },
        }
        walk_forward = {
            "status": "ok",
            "aggregate": {
                "window_count": 7,
                "pass_rate": 0.25,
                "delta_summary": {
                    "average_strategy_return": {
                        "mean": -0.01,
                        "standard_error_across_windows": 0.005,
                    },
                    "p05_strategy_return": {
                        "mean": 0.01,
                        "standard_error_across_windows": 0.002,
                    },
                    "average_strategy_max_loss": {
                        "mean": 0.01,
                        "standard_error_across_windows": 0.002,
                    },
                    "whipsaw_rate": {
                        "mean": 0.0,
                        "standard_error_across_windows": 0.002,
                    },
                },
            },
        }
        diagnostics = _combine_promotion_diagnostics(single, walk_forward)
        self.assertFalse(diagnostics["pass"])
        self.assertFalse(diagnostics["criteria"]["walk_forward_pass"])


if __name__ == "__main__":
    unittest.main()
