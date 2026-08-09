import unittest
from datetime import date

from tpsl.cli import _build_parser


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


if __name__ == "__main__":
    unittest.main()
