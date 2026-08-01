import unittest

import pandas as pd

from tpsl.backtest import calculate_backtest_metrics, simulate_next_day


class BacktestTests(unittest.TestCase):
    def test_dual_hit_uses_stop_first(self) -> None:
        result = simulate_next_day(
            close_price=10.0,
            take_profit_price=10.3,
            stop_trigger_price=9.8,
            stop_limit_price=9.7,
            next_open=10.0,
            next_high=10.5,
            next_low=9.6,
            next_close=10.2,
            next_is_suspended=False,
            stop_order_type="limit",
        )
        self.assertEqual(result["outcome"], "DUAL_HIT_STOP_FIRST")
        self.assertEqual(result["exit_price"], 9.7)
        self.assertEqual(result["dual_hit"], 1)

    def test_gap_limit_can_remain_unfilled(self) -> None:
        result = simulate_next_day(
            close_price=10.0,
            take_profit_price=10.3,
            stop_trigger_price=9.8,
            stop_limit_price=9.7,
            next_open=9.4,
            next_high=9.6,
            next_low=9.1,
            next_close=9.3,
            next_is_suspended=False,
            stop_order_type="limit",
        )
        self.assertEqual(result["outcome"], "STOP_LIMIT_UNFILLED")
        self.assertEqual(result["exit_price"], 9.3)
        self.assertEqual(result["stop_limit_unfilled"], 1)

    def test_no_take_profit_still_enforces_stop(self) -> None:
        result = simulate_next_day(
            close_price=10.0,
            take_profit_price=None,
            stop_trigger_price=9.8,
            stop_limit_price=9.7,
            next_open=10.0,
            next_high=11.0,
            next_low=9.6,
            next_close=10.5,
            next_is_suspended=False,
            stop_order_type="limit",
        )
        self.assertEqual(result["outcome"], "STOP_LOSS")
        self.assertEqual(result["exit_price"], 9.7)

    def test_metrics_include_drawdown_and_profit_factor(self) -> None:
        trades = pd.DataFrame(
            {
                "signal_date": [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ],
                "return_pct": [0.02, -0.01, 0.01],
                "outcome": ["TAKE_PROFIT", "STOP_LOSS", "NO_TRIGGER_CLOSE"],
                "dual_hit": [0, 0, 0],
                "gap_stop": [0, 0, 0],
                "stop_limit_unfilled": [0, 0, 0],
            }
        )
        metrics = calculate_backtest_metrics(trades)
        self.assertEqual(metrics["trade_count"], 3)
        self.assertAlmostEqual(metrics["profit_factor"], 3.0)
        self.assertLessEqual(metrics["maximum_drawdown"], 0)


if __name__ == "__main__":
    unittest.main()
