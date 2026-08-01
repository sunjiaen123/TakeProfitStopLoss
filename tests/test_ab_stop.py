import unittest

import pandas as pd

from tpsl.ab_stop import summarize_ab_results
from tpsl.config import RecommendationConfig
from tpsl.risk import make_risk_decision


def _row(path_id: int, strategy_return: float, stopped: int, fly: bool) -> dict:
    exit_price = 10.0 if stopped else None
    return {
        "path_id": path_id,
        "entry_date": pd.Timestamp("2026-01-01").date(),
        "symbol": "000001.SZ",
        "holding_days": 20,
        "strategy": "A_pure_2pct",
        "stop_order_type": "limit",
        "atr_multiplier": 0.0,
        "limit_slippage_pct": 0.003,
        "minimum_stop_gap_pct": 0.02,
        "average_minimum_stop_gap_pct": 0.02,
        "min_minimum_stop_gap_pct": 0.02,
        "max_minimum_stop_gap_pct": 0.02,
        "entry_price": 10.0,
        "exit_date": pd.Timestamp("2026-01-02").date() if stopped else None,
        "exit_price": exit_price,
        "exit_outcome": "STOP_TRIGGERED" if stopped else "HOLD_TO_HORIZON",
        "stopped": stopped,
        "strategy_return": strategy_return,
        "hold_return": 0.0,
        "excess_return": strategy_return,
        "strategy_max_drawdown": min(0.0, strategy_return),
        "hold_max_drawdown": -0.05,
        "average_drawdown_reduction": 0.0,
        "strategy_max_loss": min(0.0, strategy_return),
        "hold_max_loss": -0.05,
        "average_max_loss_reduction": 0.0,
        "stop_update_count": 1,
        "stop_limit_unfilled_count": 0,
        "loss_avoided": 0,
        "whipsaw": 0,
        "exit_day": 1 if stopped else 20,
        "post_exit_high_close_5d": 10.6 if fly else 10.1,
        "post_exit_high_close_10d": 10.6 if fly else 10.1,
        "post_exit_high_close_20d": 10.6 if fly else 10.1,
    }


class AbStopTests(unittest.TestCase):
    def test_atr_multiplier_zero_neutralizes_model_and_atr_to_pure_2pct(self) -> None:
        config = RecommendationConfig(
            atr_stop_multiplier=0.0,
            minimum_stop_gap_pct=0.02,
            stop_limit_slippage_pct=0.003,
        )
        decision = make_risk_decision(
            close=100.0,
            avg_cost=100.0,
            max_loss_pct=0.05,
            current_stop=None,
            predicted_high_return=0.03,
            predicted_low_return=-0.20,
            similar_high_return=0.02,
            similar_low_return=-0.15,
            atr_14=10.0,
            similarity_distance=0.1,
            sample_count=100,
            config=config,
        )

        self.assertEqual(decision.stop_trigger_price, 98.0)

    def test_sell_fly_all_paths_denominator(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, -0.01, 1, True),
                _row(2, 0.02, 1, False),
                _row(3, 0.03, 0, False),
                _row(4, -0.02, 0, False),
            ]
        )
        summary = summarize_ab_results(frame)

        self.assertAlmostEqual(summary["sell_fly_exited_10d_5pct"], 0.5)
        self.assertAlmostEqual(summary["sell_fly_all_10d_5pct"], 0.25)


if __name__ == "__main__":
    unittest.main()
