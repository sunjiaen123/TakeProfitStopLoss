import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tpsl.volatility_stop import (
    VOLATILITY_STOP_GAP_COLUMN,
    VOLATILITY_SIGMA_USED_COLUMN,
    _calibrate_k_to_target_gap,
    _calibrate_k_to_target_stop_ratio,
    _select_by_constraint,
    add_volatility_stop_gaps,
)


def _config(**overrides):
    volatility = {
        "lookback_days": 120,
        "k": 1.5,
        "k_values": (0.5, 1.0, 1.5, 2.0, 3.0),
        "shrink_n0": 120.0,
        "gap_min": 0.005,
        "gap_max": 0.10,
        "pool": "industry",
        "adjustment_factor": 1.0,
        "adjustment_min": 0.8,
        "adjustment_max": 1.2,
    }
    volatility.update(overrides)
    return SimpleNamespace(volatility_stop=SimpleNamespace(**volatility))


def _candidate_row(value: float, strategy_return: float, max_loss: float):
    hold_return = 0.01
    return {
        "k": value,
        "stopped": 1,
        "stop_limit_unfilled_count": 0,
        "strategy_return": strategy_return,
        "hold_return": hold_return,
        "excess_return": strategy_return - hold_return,
        "strategy_max_drawdown": max_loss,
        "hold_max_drawdown": -0.06,
        "strategy_max_loss": max_loss,
        "hold_max_loss": -0.06,
        "loss_avoided": int(strategy_return > hold_return),
        "whipsaw": 0,
    }


class VolatilityStopTests(unittest.TestCase):
    def test_add_volatility_stop_gaps_uses_shrunk_atr(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "symbol": "A",
                    "trade_date": "2026-01-01",
                    "industry": "I1",
                    "atr_14_pct": 0.02,
                },
                {
                    "symbol": "B",
                    "trade_date": "2026-01-01",
                    "industry": "I1",
                    "atr_14_pct": 0.06,
                },
            ]
        )
        result = add_volatility_stop_gaps(frame, _config(shrink_n0=0), k=2.0)
        self.assertTrue(np.isfinite(result[VOLATILITY_SIGMA_USED_COLUMN]).all())
        gap_a = float(result.loc[result["symbol"] == "A", VOLATILITY_STOP_GAP_COLUMN].iloc[0])
        gap_b = float(result.loc[result["symbol"] == "B", VOLATILITY_STOP_GAP_COLUMN].iloc[0])
        self.assertAlmostEqual(gap_a, 0.04)
        self.assertAlmostEqual(gap_b, 0.10)

    def test_select_by_constraint_prefers_safer_loss_with_return_constraint(self) -> None:
        frame = pd.DataFrame(
            [
                _candidate_row(1.0, 0.010, -0.03),
                _candidate_row(2.0, 0.009, -0.02),
                _candidate_row(3.0, 0.000, -0.01),
            ]
        )
        selection = _select_by_constraint(
            frame,
            value_column="k",
            tolerance=0.002,
        )
        self.assertEqual(selection["selected_value"], 2.0)
        self.assertTrue(selection["selected"]["return_constraint_pass"])

    def test_select_by_constraint_prefers_wider_when_loss_is_near_tied(self) -> None:
        frame = pd.DataFrame(
            [
                _candidate_row(1.0, 0.010, -0.020),
                _candidate_row(2.0, 0.010, -0.021),
            ]
        )
        selection = _select_by_constraint(
            frame,
            value_column="k",
            tolerance=0.002,
            max_loss_tie_tolerance=0.002,
        )
        self.assertEqual(selection["selected_value"], 2.0)

    def test_calibrate_k_to_target_gap_uses_average_gap_not_returns(self) -> None:
        rows = pd.DataFrame(
            {
                VOLATILITY_SIGMA_USED_COLUMN: [0.01, 0.02, 0.03],
            }
        )
        paths = [{"rows": rows}]
        selection = _calibrate_k_to_target_gap(
            paths,
            {0},
            _config(gap_min=0.0, gap_max=1.0),
            target_gap=0.04,
        )
        self.assertAlmostEqual(selection["selected_k"], 2.0, places=5)
        self.assertAlmostEqual(
            selection["achieved_train_average_gap_pct"],
            0.04,
            places=5,
        )

    def test_calibrate_k_to_target_stop_ratio_uses_average_multiple(self) -> None:
        rows = pd.DataFrame(
            {
                VOLATILITY_SIGMA_USED_COLUMN: [0.01, 0.02, 0.04],
            }
        )
        paths = [{"rows": rows}]
        selection = _calibrate_k_to_target_stop_ratio(
            paths,
            {0},
            _config(gap_min=0.0, gap_max=1.0),
            fixed_gap=0.02,
        )
        expected = np.mean([2.0, 1.0, 0.5])
        self.assertAlmostEqual(selection["selected_k"], expected, places=5)
        self.assertAlmostEqual(
            selection["achieved_train_average_stop_ratio"],
            expected,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
