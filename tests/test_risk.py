import unittest

from tpsl.config import RecommendationConfig
from tpsl.risk import make_risk_decision, round_to_tick


class RiskTests(unittest.TestCase):
    def test_round_to_tick(self) -> None:
        self.assertEqual(round_to_tick(10.127, 0.01, "down"), 10.12)
        self.assertEqual(round_to_tick(10.127, 0.01, "up"), 10.13)

    def test_existing_stop_is_not_lowered(self) -> None:
        decision = make_risk_decision(
            close=10.0,
            avg_cost=9.5,
            max_loss_pct=0.05,
            current_stop=9.7,
            predicted_high_return=0.04,
            predicted_low_return=-0.03,
            similar_high_return=0.03,
            similar_low_return=-0.025,
            atr_14=0.15,
            similarity_distance=0.2,
            sample_count=100,
            config=RecommendationConfig(),
        )
        self.assertTrue(decision.take_profit_enabled)
        self.assertGreaterEqual(decision.stop_trigger_price, 9.70)
        self.assertGreater(decision.take_profit_price, 10.0)
        self.assertLess(decision.stop_limit_price, decision.stop_trigger_price)

    def test_minimum_gap_does_not_override_cost_loss_floor(self) -> None:
        decision = make_risk_decision(
            close=97.0,
            avg_cost=100.0,
            max_loss_pct=0.05,
            current_stop=None,
            predicted_high_return=0.04,
            predicted_low_return=-0.08,
            similar_high_return=0.03,
            similar_low_return=-0.08,
            atr_14=4.0,
            similarity_distance=0.2,
            sample_count=100,
            config=RecommendationConfig(minimum_stop_gap_pct=0.05),
        )
        self.assertEqual(decision.stop_trigger_price, 95.0)

    def test_breached_cost_floor_uses_immediate_close_trigger(self) -> None:
        decision = make_risk_decision(
            close=94.0,
            avg_cost=100.0,
            max_loss_pct=0.05,
            current_stop=None,
            predicted_high_return=0.04,
            predicted_low_return=-0.08,
            similar_high_return=0.03,
            similar_low_return=-0.08,
            atr_14=4.0,
            similarity_distance=0.2,
            sample_count=100,
            config=RecommendationConfig(minimum_stop_gap_pct=0.05),
        )
        self.assertEqual(decision.stop_trigger_price, 94.0)

    def test_take_profit_is_above_cost_floor(self) -> None:
        decision = make_risk_decision(
            close=9.8,
            avg_cost=10.0,
            max_loss_pct=0.05,
            current_stop=None,
            predicted_high_return=0.005,
            predicted_low_return=-0.02,
            similar_high_return=0.004,
            similar_low_return=-0.025,
            atr_14=0.20,
            similarity_distance=0.2,
            sample_count=100,
            config=RecommendationConfig(),
        )
        self.assertFalse(decision.take_profit_enabled)
        self.assertIsNone(decision.take_profit_price)
        self.assertIn("成本保护线", decision.reason)

    def test_low_risk_reward_disables_take_profit_but_keeps_stop(self) -> None:
        decision = make_risk_decision(
            close=10.0,
            avg_cost=10.0,
            max_loss_pct=0.05,
            current_stop=None,
            predicted_high_return=0.015,
            predicted_low_return=-0.02,
            similar_high_return=0.012,
            similar_low_return=-0.025,
            atr_14=0.20,
            similarity_distance=0.1,
            sample_count=100,
            config=RecommendationConfig(minimum_risk_reward=2.0),
        )
        self.assertFalse(decision.take_profit_enabled)
        self.assertIsNone(decision.take_profit_price)
        self.assertGreater(decision.stop_trigger_price, 0)
        self.assertIn("盈亏比不足", decision.take_profit_reason)


if __name__ == "__main__":
    unittest.main()
