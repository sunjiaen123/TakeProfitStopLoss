import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tpsl.exit_machine import (
    _recent_confirmed_swing_lows,
    _simulate_exit_machine_path,
    _simulate_fixed_baseline_path,
    summarize_exit_results,
)


def _config():
    return SimpleNamespace(
        exit_machine=SimpleNamespace(
            sell_fly_days=(5, 10, 20),
            sell_fly_thresholds=(0.05, 0.10),
            winner_peak_gain=0.15,
            trend_capture_ratio=0.5,
            n_init=8,
        )
    )


def _simulation_config():
    return SimpleNamespace(
        recommendation=SimpleNamespace(price_tick=0.01),
        positions=SimpleNamespace(default_max_loss_pct=0.05),
        exit_machine=SimpleNamespace(
            sell_fly_days=(5, 10, 20),
            fixed_baseline_gap=0.02,
            r_floor_atr_mult=0.5,
            progress_r=1.0,
            chandelier_mult=3.0,
            swing_trend_buffer=0.3,
            breakdown_buffer=0.2,
            profit_tier_1_gain=0.08,
            profit_tier_1_floor_pct=0.005,
            profit_tier_2_gain=0.15,
            profit_tier_2_capture=0.30,
            profit_tier_3_gain=0.30,
            profit_tier_3_capture=0.45,
            stop_order_type="limit",
            limit_slippage_pct=0.003,
        ),
    )


def _path_from_rows(rows: pd.DataFrame) -> dict:
    return {
        "entry_date": pd.Timestamp("2026-01-01").date(),
        "symbol": "000001.SZ",
        "holding_days": len(rows),
        "entry_price": float(rows.iloc[0]["raw_close"]),
        "rows": rows,
    }


def _rows_from_closes(
    closes: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    ma20: float = 10.0,
    swing_low: float = 9.5,
) -> pd.DataFrame:
    highs = highs or [max(10.05, close) for close in closes]
    lows = lows or [min(9.90, close) for close in closes]
    dates = pd.date_range("2026-01-01", periods=len(closes) + 1, freq="D")
    rows = []
    for index, close in enumerate(closes):
        next_close = closes[index + 1] if index + 1 < len(closes) else close
        rows.append(
            {
                "raw_close": close,
                "raw_high": highs[index],
                "raw_low": lows[index],
                "atr_14_pct": 0.01,
                "ma_10": ma20,
                "ma_20": ma20,
                "recent_swing_low": swing_low,
                "next_open": next_close,
                "next_high": highs[index + 1] if index + 1 < len(highs) else next_close,
                "next_low": lows[index + 1] if index + 1 < len(lows) else next_close,
                "next_close": next_close,
                "execution_date": dates[index + 1],
                "next_is_suspended": False,
            }
        )
    return pd.DataFrame(rows)


def _row(path_id: int, strategy_return: float, stopped: int, fly: bool):
    exit_price = 10.0
    return {
        "path_id": path_id,
        "strategy": "exit_machine",
        "stopped": stopped,
        "stop_limit_unfilled_count": 0,
        "strategy_return": strategy_return,
        "hold_return": 0.0,
        "excess_return": strategy_return,
        "strategy_max_drawdown": min(0.0, strategy_return),
        "hold_max_drawdown": -0.05,
        "strategy_max_loss": min(0.0, strategy_return),
        "hold_max_loss": -0.05,
        "loss_avoided": 0,
        "whipsaw": 0,
        "exit_day": 3,
        "exit_price": exit_price,
        "held_peak_gain": max(strategy_return, 0.01),
        "full_path_peak_gain": 0.20,
        "full_path_return": -0.01,
        "post_exit_high_close_5d": exit_price * (1.06 if fly else 1.01),
        "post_exit_high_close_10d": exit_price * (1.06 if fly else 1.01),
        "post_exit_high_close_20d": exit_price * (1.06 if fly else 1.01),
    }


class ExitMachineTests(unittest.TestCase):
    def test_recent_swing_low_is_available_only_after_confirmation(self) -> None:
        lows = pd.Series([10.0, 9.0, 8.0, 9.0, 10.0, 7.0, 8.0, 9.0])
        recent = _recent_confirmed_swing_lows(lows, pivot_k=2)
        self.assertTrue(np.isnan(recent.iloc[3]))
        self.assertEqual(float(recent.iloc[4]), 8.0)
        self.assertEqual(float(recent.iloc[7]), 7.0)

    def test_sell_fly_reports_all_paths_denominator(self) -> None:
        frame = pd.DataFrame(
            [
                _row(1, -0.01, 1, True),
                _row(2, 0.02, 1, False),
                _row(3, 0.03, 0, False),
                _row(4, -0.02, 0, False),
            ]
        )
        summary = summarize_exit_results(frame, _config())
        self.assertAlmostEqual(summary["sell_fly_exited_10d_5pct"], 0.5)
        self.assertAlmostEqual(summary["sell_fly_all_10d_5pct"], 0.25)

    def test_v2_matches_fixed_baseline_before_profit_gate(self) -> None:
        rows = _rows_from_closes([10.0] + [9.95] * 11)
        path = _path_from_rows(rows)
        config = _simulation_config()

        machine = _simulate_exit_machine_path(path_id=1, path=path, config=config)
        baseline = _simulate_fixed_baseline_path(path_id=1, path=path, config=config)

        self.assertEqual(machine["exit_reason"], baseline["exit_reason"])
        self.assertEqual(machine["exit_day"], baseline["exit_day"])
        self.assertAlmostEqual(machine["exit_price"], baseline["exit_price"])
        self.assertAlmostEqual(machine["strategy_return"], baseline["strategy_return"])
        self.assertAlmostEqual(machine["strategy_max_loss"], baseline["strategy_max_loss"])

    def test_v2_requires_stop_above_cost_before_trend_exit(self) -> None:
        closes = [10.0, 10.05, 10.00, 9.98, 9.97]
        highs = [10.0, 10.30, 10.04, 10.02, 10.00]
        lows = [10.0, 10.00, 9.90, 9.90, 9.90]
        rows = _rows_from_closes(
            closes,
            highs=highs,
            lows=lows,
            ma20=11.0,
            swing_low=10.20,
        )
        path = _path_from_rows(rows)
        config = _simulation_config()

        machine = _simulate_exit_machine_path(path_id=1, path=path, config=config)
        baseline = _simulate_fixed_baseline_path(path_id=1, path=path, config=config)

        self.assertNotEqual(machine["exit_reason"], "technical_exit")
        self.assertEqual(machine["exit_reason"], baseline["exit_reason"])
        self.assertEqual(machine["exit_day"], baseline["exit_day"])


if __name__ == "__main__":
    unittest.main()
