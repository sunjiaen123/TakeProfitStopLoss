import unittest

import pandas as pd

from tpsl.stop_tuning import (
    build_parameter_grid,
    rank_results,
    summarize_paths,
)


class StopTuningTests(unittest.TestCase):
    def test_grid_contains_limit_and_market(self) -> None:
        grid = build_parameter_grid(
            (1.0, 2.0),
            (0.003, 0.006),
            (0.005, 0.02),
            True,
        )
        self.assertEqual(len(grid), 12)
        self.assertEqual(
            sum(item["stop_order_type"] == "market" for item in grid),
            4,
        )

    def test_summary_and_ranking(self) -> None:
        paths = pd.DataFrame(
            {
                "stopped": [1, 0, 1],
                "stop_limit_unfilled_count": [0, 0, 1],
                "strategy_return": [-0.02, 0.03, -0.01],
                "hold_return": [-0.05, 0.04, 0.02],
                "excess_return": [0.03, -0.01, -0.03],
                "strategy_max_drawdown": [-0.03, -0.02, -0.04],
                "hold_max_drawdown": [-0.08, -0.03, -0.05],
                "strategy_max_loss": [-0.02, -0.01, -0.03],
                "hold_max_loss": [-0.07, -0.02, -0.04],
                "loss_avoided": [1, 0, 0],
                "whipsaw": [0, 0, 1],
            }
        )
        summary = summarize_paths(paths)
        self.assertEqual(summary["path_count"], 3)
        self.assertGreater(summary["average_max_loss_reduction"], 0)

        results = pd.DataFrame(
            [
                {
                    "holding_days": 0,
                    "p05_strategy_return": -0.03,
                    "average_strategy_max_loss": -0.02,
                    "average_excess_return": 0.01,
                    "stopped_rate": 0.5,
                    "stop_limit_unfilled_rate": 0.0,
                    "whipsaw_rate": 0.1,
                },
                {
                    "holding_days": 0,
                    "p05_strategy_return": -0.05,
                    "average_strategy_max_loss": -0.04,
                    "average_excess_return": -0.01,
                    "stopped_rate": 1.0,
                    "stop_limit_unfilled_rate": 0.1,
                    "whipsaw_rate": 0.2,
                },
            ]
        )
        ranked = rank_results(results)
        self.assertEqual(int(ranked.iloc[0]["rank_no"]), 1)
        self.assertEqual(float(ranked.iloc[0]["average_excess_return"]), 0.01)


if __name__ == "__main__":
    unittest.main()
