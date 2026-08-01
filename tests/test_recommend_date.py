from datetime import date
import unittest

from sqlalchemy import create_engine, text

from tpsl.cli import _parse_recommend_date, _parse_sync_end_date
from tpsl.config import load_config
from tpsl.pipeline import (
    latest_complete_position_bar_date,
    resolve_recommend_as_of_date,
)


class RecommendDateTests(unittest.TestCase):
    def test_latest_uses_date_complete_for_all_positions(self) -> None:
        config = load_config("config.toml")
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE stock_positions (
                        symbol TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        avg_cost REAL NOT NULL,
                        entry_date DATE NULL,
                        max_loss_pct REAL NULL,
                        current_stop REAL NULL,
                        status TEXT NOT NULL
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE TABLE stock_daily_bars (
                        symbol TEXT NOT NULL,
                        trade_date DATE NOT NULL
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO stock_positions
                    (symbol, quantity, avg_cost, status)
                    VALUES
                    ('000001.SZ', 100, 10, 'HOLDING'),
                    ('000002.SZ', 100, 20, 'HOLDING')
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO stock_daily_bars
                    (symbol, trade_date)
                    VALUES
                    ('000001.SZ', '2026-01-02'),
                    ('000002.SZ', '2026-01-02'),
                    ('000001.SZ', '2026-01-05')
                    """
                )
            )

        self.assertEqual(
            latest_complete_position_bar_date(engine, config),
            date(2026, 1, 2),
        )
        self.assertEqual(
            resolve_recommend_as_of_date(engine, config, "latest"),
            date(2026, 1, 2),
        )
        self.assertEqual(_parse_recommend_date("latest"), "latest")
        self.assertEqual(
            _parse_recommend_date("2026-01-05"),
            date(2026, 1, 5),
        )
        self.assertEqual(_parse_sync_end_date("2026-01-05"), date(2026, 1, 5))
        self.assertEqual(_parse_sync_end_date("today"), date.today())


if __name__ == "__main__":
    unittest.main()
