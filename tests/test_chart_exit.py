import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tpsl.chart_exit import (
    _confirmed_swing_highs,
    _initial_stop,
    _profit_floor,
    _reentry_signal,
    _simulate_hold,
    _simulate_chart,
    _simulate_one_path,
    add_chart_exit_indicators,
    recommend_chart_exit,
)
from tpsl.config import ChartExitConfig, PositionsConfig, RecommendationConfig


def _config():
    chart = SimpleNamespace(
        initial_min_gap_pct=0.025,
        entry_structure_buffer=0.3,
        r_floor_atr_mult=0.5,
        profit_tier_1_gain=0.08,
        profit_tier_1_floor_pct=0.005,
        profit_tier_2_gain=0.15,
        profit_tier_2_capture=0.50,
        profit_tier_3_gain=0.30,
        profit_tier_3_capture=0.60,
        reentry_cooldown_days=2,
        max_entries_safety=3,
        max_reentry_risk_pct=0.05,
        commission_rate=0.0003,
        stamp_tax_rate=0.0005,
        market_slippage_pct=0.001,
        winner_peak_gain=0.15,
        trend_capture_ratio=0.5,
        sell_fly_days=(5,),
        sell_fly_thresholds=(0.05,),
    )
    return SimpleNamespace(
        chart_exit=chart,
        positions=SimpleNamespace(default_max_loss_pct=0.05),
        recommendation=SimpleNamespace(price_tick=0.01),
    )


class ChartExitTests(unittest.TestCase):
    def test_swing_high_is_available_only_after_confirmation(self) -> None:
        highs = pd.Series([8.0, 9.0, 10.0, 9.0, 8.0, 11.0, 10.0, 9.0])
        recent = _confirmed_swing_highs(highs, pivot_k=2)
        self.assertTrue(np.isnan(recent.iloc[3]))
        self.assertEqual(float(recent.iloc[4]), 10.0)
        self.assertEqual(float(recent.iloc[7]), 11.0)

    def test_initial_stop_keeps_2_5_to_5_pct_breathing_room(self) -> None:
        config = _config()
        tight_structure = pd.Series(
            {"recent_swing_low": 9.95, "atr_14_pct": 0.02}
        )
        loose_structure = pd.Series(
            {"recent_swing_low": 9.20, "atr_14_pct": 0.02}
        )
        self.assertAlmostEqual(_initial_stop(10.0, tight_structure, config), 9.75)
        self.assertAlmostEqual(_initial_stop(10.0, loose_structure, config), 9.50)

    def test_profit_floor_is_staged_and_monotonic(self) -> None:
        config = _config()
        self.assertTrue(np.isneginf(_profit_floor(10.0, 10.70, config)))
        self.assertAlmostEqual(_profit_floor(10.0, 11.00, config), 10.05)
        self.assertAlmostEqual(_profit_floor(10.0, 12.00, config), 11.00)
        self.assertAlmostEqual(_profit_floor(10.0, 14.00, config), 12.40)

    def test_reentry_is_signal_driven_not_fixed_once(self) -> None:
        config = _config()
        row = pd.Series(
            {
                "raw_close": 11.0,
                "ma_trend": 10.0,
                "ma_long": 9.5,
                "ma_trend_slope": 0.01,
                "prior_breakout_high": 10.8,
            }
        )
        last_exit = {
            "exit_day_index": 2,
            "exit_candle_high": 10.7,
            "exit_reason": "ma_confirmed_breakdown",
        }
        self.assertTrue(
            _reentry_signal(
                row,
                state={"entries": 1},
                last_exit=last_exit,
                current_index=4,
                config=config,
            )
        )
        self.assertTrue(
            _reentry_signal(
                row,
                state={"entries": 2},
                last_exit=last_exit,
                current_index=4,
                config=config,
            )
        )
        self.assertFalse(
            _reentry_signal(
                row,
                state={"entries": 3},
                last_exit=last_exit,
                current_index=4,
                config=config,
            )
        )

    def test_hold_result_deducts_round_trip_costs(self) -> None:
        dates = pd.date_range("2026-01-01", periods=2, freq="D")
        rows = pd.DataFrame(
            [
                {
                    "next_close": 10.0,
                    "next_low": 9.8,
                    "next_high": 10.2,
                    "execution_date": dates[1],
                }
            ]
        )
        path = {
            "entry_date": dates[0].date(),
            "symbol": "000001.SZ",
            "holding_days": 1,
            "entry_price": 10.0,
            "rows": rows,
            "full_rows": rows,
        }
        result, _ = _simulate_hold(1, path, _config())
        self.assertLess(result["strategy_return"], 0.0)
        self.assertGreater(result["transaction_cost_rate"], 0.0)

    def test_all_five_strategies_run_on_the_same_path(self) -> None:
        dates = pd.date_range("2026-01-01", periods=16, freq="D")
        full_rows = []
        for index in range(15):
            close = 10.0 + index * 0.03
            next_close = 10.0 + (index + 1) * 0.03
            full_rows.append(
                {
                    "raw_close": close,
                    "raw_high": close + 0.08,
                    "raw_low": close - 0.08,
                    "atr_14_pct": 0.02,
                    "ma_fast": close - 0.03,
                    "ma_trend": close - 0.05,
                    "ma_long": close - 0.10,
                    "ma_trend_slope": 0.01,
                    "recent_swing_low": close - 0.30,
                    "prior_breakout_high": close - 0.01,
                    "predicted_high_return": 0.03,
                    "predicted_low_return": -0.02,
                    "median_high_return": 0.03,
                    "low_return_q20": -0.02,
                    "mean_distance": 0.5,
                    "sample_count": 100,
                    "next_open": next_close,
                    "next_high": next_close + 0.08,
                    "next_low": next_close - 0.08,
                    "next_close": next_close,
                    "execution_date": dates[index + 1],
                    "next_is_suspended": False,
                }
            )
        full_frame = pd.DataFrame(full_rows)
        path = {
            "entry_date": dates[0].date(),
            "symbol": "000001.SZ",
            "holding_days": 10,
            "entry_price": 10.0,
            "rows": full_frame.iloc[:10].reset_index(drop=True),
            "full_rows": full_frame,
        }
        config = SimpleNamespace(
            chart_exit=ChartExitConfig(
                sell_fly_days=(5,),
                sell_fly_thresholds=(0.05,),
            ),
            positions=PositionsConfig(),
            recommendation=RecommendationConfig(minimum_stop_gap_pct=0.02),
        )
        records, ledger = _simulate_one_path(1, path, config)
        self.assertEqual(len(records), 5)
        self.assertEqual(len({record["strategy"] for record in records}), 5)
        self.assertGreaterEqual(len(ledger), 5)
        self.assertTrue(all(np.isfinite(record["strategy_return"]) for record in records))
        self.assertTrue(all(0.0 <= record["profit_giveback"] <= 1.0 for record in records))

    def test_v2_does_not_exit_on_normal_lower_shadow_or_upper_wick(self) -> None:
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        rows = []
        closes = [10.0, 10.1, 10.2, 10.3]
        for index, close in enumerate(closes):
            next_close = closes[min(index + 1, len(closes) - 1)]
            rows.append(
                {
                    "raw_close": close,
                    "raw_high": 12.0,
                    "raw_low": 9.6,
                    "atr_14_pct": 0.02,
                    "ma_fast": 9.9,
                    "ma_trend": 9.8,
                    "ma_long": 9.7,
                    "ma_trend_slope": 0.01,
                    "recent_swing_low": 9.8,
                    "prior_breakout_high": 11.0,
                    "next_open": next_close,
                    "next_high": 12.0,
                    "next_low": 9.6,
                    "next_close": next_close,
                    "execution_date": dates[index + 1],
                    "next_is_suspended": False,
                }
            )
        frame = pd.DataFrame(rows)
        path = {
            "entry_date": dates[0].date(),
            "symbol": "000001.SZ",
            "holding_days": len(frame),
            "entry_price": 10.0,
            "rows": frame,
            "full_rows": frame,
        }
        config = SimpleNamespace(
            chart_exit=ChartExitConfig(
                sell_fly_days=(1,),
                sell_fly_thresholds=(0.05,),
            ),
            positions=PositionsConfig(),
            recommendation=RecommendationConfig(minimum_stop_gap_pct=0.02),
        )
        result, ledger = _simulate_chart(
            1,
            path,
            config,
            allow_reentry=False,
            hold_return=0.0,
        )
        self.assertEqual(ledger[-1]["exit_reason"], "horizon_exit")
        self.assertEqual(result["stopped"], 0)

    def test_chart_indicators_are_available_for_production_recommendation(self) -> None:
        dates = pd.date_range("2026-01-01", periods=8, freq="D")
        frame = pd.DataFrame(
            {
                "symbol": ["000001.SZ"] * len(dates),
                "trade_date": dates,
                "raw_close": [10.0, 10.1, 10.2, 10.1, 10.3, 10.4, 10.2, 10.5],
                "raw_high": [10.2, 10.3, 10.4, 10.3, 10.5, 10.6, 10.4, 10.7],
                "raw_low": [9.8, 9.9, 10.0, 9.9, 10.1, 10.2, 10.0, 10.3],
            }
        )
        config = SimpleNamespace(
            chart_exit=ChartExitConfig(
                ma_fast=3,
                ma_trend=4,
                ma_long=5,
                ma_slope_lookback=1,
                reentry_breakout_lookback=3,
            )
        )
        output = add_chart_exit_indicators(frame, config)
        for column in (
            "ma_fast",
            "ma_trend",
            "ma_long",
            "ma_trend_slope",
            "recent_swing_low",
            "prior_breakout_high",
        ):
            self.assertIn(column, output.columns)
        self.assertTrue(np.isfinite(output.iloc[-1]["ma_fast"]))

    def test_recommend_chart_exit_emits_s3_initial_failure(self) -> None:
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        closes = [10.0, 9.95, 9.90, 9.85, 9.80]
        history = pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "trade_date": dates[index],
                    "raw_close": close,
                    "raw_high": close + 0.1,
                    "raw_low": close - 0.1,
                    "atr_14_pct": 0.02,
                    "ma_fast": close + 0.05,
                    "ma_trend": 10.5,
                    "ma_long": 10.4,
                    "ma_trend_slope": -0.01,
                    "recent_swing_low": 9.0,
                }
                for index, close in enumerate(closes)
            ]
        )
        config = SimpleNamespace(
            chart_exit=ChartExitConfig(
                enabled=True,
                production_profile="S3_fast_scale_90",
                position_scale=0.90,
                initial_days=5,
                breakdown_confirm_closes=1,
            ),
            positions=PositionsConfig(default_max_loss_pct=0.05),
            recommendation=RecommendationConfig(price_tick=0.01),
        )
        decision = recommend_chart_exit(
            history,
            {
                "symbol": "000001.SZ",
                "entry_date": dates[0].date(),
                "avg_cost": 10.0,
                "max_loss_pct": 0.05,
            },
            dates[-1].date(),
            config,
        )
        self.assertEqual(decision.action, "EXIT_NEXT_OPEN")
        self.assertEqual(decision.reason, "initial_failure")
        self.assertEqual(decision.profile, "S3_fast_scale_90")
        self.assertAlmostEqual(decision.position_scale, 0.90)
        self.assertEqual(decision.trade_days, 5)
        self.assertAlmostEqual(decision.stop_trigger_price, 9.50)
        self.assertAlmostEqual(decision.stop_limit_price, 9.47)


if __name__ == "__main__":
    unittest.main()
