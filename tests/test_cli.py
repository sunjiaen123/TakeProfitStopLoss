import unittest
from contextlib import redirect_stdout
from datetime import date
from io import StringIO

import pandas as pd

from tpsl.cli import (
    _build_parser,
    _print_recommendations,
    _require_complete_daily_sync,
)
from tpsl.pipeline import recommendation_preview_frame


class CliTests(unittest.TestCase):
    def test_daily_defaults_to_today_dry_run_output(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--config", "config.toml", "daily"])
        self.assertEqual(args.command, "daily")
        self.assertEqual(args.config, "config.toml")
        self.assertEqual(args.end_date, date.today())
        self.assertEqual(args.output, "output/current_stops.csv")
        self.assertFalse(args.write_db)

    def test_daily_write_db_is_explicit(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--config",
                "config.toml",
                "daily",
                "--end-date",
                "2026-08-03",
                "--output",
                "output/custom.csv",
                "--write-db",
            ]
        )
        self.assertEqual(args.end_date, date(2026, 8, 3))
        self.assertEqual(args.output, "output/custom.csv")
        self.assertTrue(args.write_db)

    def test_s3_output_does_not_print_legacy_strategy_fields(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "close_price": 10.0,
                    "avg_cost": 9.5,
                    "strategy_profile": "S3_fast_scale_90",
                    "chart_exit_action": "HOLD",
                    "chart_exit_reason": "chart_hold",
                    "chart_exit_stop_trigger_price": 9.02,
                    "chart_exit_stop_limit_price": 8.99,
                    "chart_exit_position_scale": 0.9,
                    "chart_exit_trend_active": 1,
                    "chart_exit_trade_days": 8,
                    "risk_reward_ratio": 99.0,
                    "confidence": 0.99,
                    "take_profit_price": 12.0,
                    "stop_trigger_price": 9.8,
                }
            ]
        )
        buffer = StringIO()
        with redirect_stdout(buffer):
            _print_recommendations(frame)
        output = buffer.getvalue()
        self.assertIn("chart_exit_action", output)
        self.assertIn("chart_exit_stop_trigger_price", output)
        self.assertNotIn("risk_reward_ratio", output)
        self.assertNotIn("confidence", output)
        self.assertNotIn("take_profit_price", output)
        self.assertNotIn(" stop_trigger_price", output)

    def test_daily_rejects_partial_position_sync(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "未计算、未写入S3建议"):
            _require_complete_daily_sync(
                {
                    "symbols": 3,
                    "successful_symbols": 2,
                    "failed_symbols": 1,
                    "errors": {"000001.SZ": "network error"},
                }
            )

    def test_daily_accepts_complete_or_already_complete_sync(self) -> None:
        _require_complete_daily_sync(
            {
                "symbols": 3,
                "attempted_symbols": 3,
                "successful_symbols": 3,
                "failed_symbols": 0,
            }
        )

    def test_s3_csv_view_keeps_operations_and_hides_diagnostics(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "as_of_date": date(2026, 9, 8),
                    "symbol": "000001.SZ",
                    "close_price": 10.0,
                    "avg_cost": 9.5,
                    "strategy_profile": "S3_fast_scale_90",
                    "chart_exit_action": "HOLD",
                    "chart_exit_reason": "chart_hold",
                    "chart_exit_stop_trigger_price": 9.02,
                    "chart_exit_stop_limit_price": 8.99,
                    "chart_exit_trend_active": 1,
                    "chart_exit_trade_days": 8,
                    "chart_exit_profit_floor_price": 9.8,
                    "chart_exit_diagnostic": "",
                    "chart_exit_ma_fast_price": 10.1,
                    "chart_exit_ma_trend_slope": 0.01,
                    "chart_exit_entry_swing_low": 9.0,
                }
            ]
        )
        preview = recommendation_preview_frame(frame)
        self.assertIn("chart_exit_action", preview.columns)
        self.assertIn("chart_exit_stop_trigger_price", preview.columns)
        self.assertNotIn("chart_exit_profit_floor_price", preview.columns)
        self.assertNotIn("chart_exit_trade_days", preview.columns)
        self.assertNotIn("chart_exit_ma_fast_price", preview.columns)
        self.assertNotIn("chart_exit_ma_trend_slope", preview.columns)
        self.assertNotIn("chart_exit_entry_swing_low", preview.columns)
        _require_complete_daily_sync(
            {
                "symbols": 3,
                "skipped_complete_symbols": 3,
                "attempted_symbols": 0,
                "successful_symbols": 0,
                "failed_symbols": 0,
            }
        )


if __name__ == "__main__":
    unittest.main()
