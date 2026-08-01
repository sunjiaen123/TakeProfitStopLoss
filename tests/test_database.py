import unittest

from tpsl.config import DatabaseConfig, load_config
from tpsl.db import _schema_path, _split_sql_statements


class DatabaseTests(unittest.TestCase):
    def test_schema_contains_all_required_tables(self) -> None:
        statements = _split_sql_statements(
            _schema_path().read_text(encoding="utf-8")
        )
        script = "\n".join(statements)
        for table in (
            "stock_master",
            "stock_positions",
            "stock_daily_bars",
            "tpsl_recommendations",
            "tpsl_model_runs",
            "tpsl_backtest_runs",
            "tpsl_backtest_trades",
            "tpsl_holding_backtest_runs",
            "tpsl_holding_backtest_positions",
            "tpsl_holding_backtest_daily",
            "tpsl_stop_tuning_runs",
            "tpsl_stop_tuning_results",
        ):
            self.assertIn(f"CREATE TABLE IF NOT EXISTS {table}", script)
        self.assertEqual(len(statements), 12)
        self.assertEqual(script.count("AUTO_INCREMENT COMMENT '自增主键'"), 12)
        self.assertNotIn("UNIQUE KEY", script)
        self.assertIn("COMMENT='股票基础信息表'", script)
        self.assertIn("dynamic_stop_gap_pct", script)
        self.assertIn("risk_model_version", script)

    def test_project_config_loads(self) -> None:
        config = load_config("config.toml")
        self.assertEqual(config.database.name, "take_profit_stop_loss")
        self.assertEqual(config.database.username, "root")
        self.assertEqual(config.database.host, "127.0.0.1")
        self.assertEqual(config.performance.workers, 10)
        self.assertEqual(config.data_sync.source, "auto")
        self.assertEqual(config.data_sync.socket_timeout_seconds, 30.0)
        self.assertEqual(config.training.max_training_symbols, 1500)
        self.assertEqual(config.backtest.max_symbols, 100)
        self.assertEqual(config.stop_tuning.holding_days, (20, 40, 60))
        self.assertEqual(
            config.stop_tuning.minimum_stop_gap_pcts,
            (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10),
        )
        self.assertEqual(config.risk_fit.max_symbols, 300)
        self.assertEqual(config.risk_fit.validation_months, 2)
        self.assertEqual(
            config.risk_fit.candidate_stop_gap_pcts,
            (0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10),
        )
        self.assertEqual(
            config.database.bar_columns["adj_factor"],
            "adj_factor",
        )
        self.assertEqual(
            config.database.bar_columns["is_suspended"],
            "is_suspended",
        )

    def test_password_special_characters_are_encoded(self) -> None:
        config = DatabaseConfig(password="a@b:c/%")
        self.assertIn("a%40b%3Ac%2F%25", config.url)


if __name__ == "__main__":
    unittest.main()
