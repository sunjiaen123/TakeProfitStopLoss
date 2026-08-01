import unittest
import math

from tpsl.data_sync import (
    _board_for,
    _format_login_error,
    _normalize_akshare_hist,
    _is_fatal_login_error,
    _sanitize_records,
    _price_limit,
    from_akshare_code,
    from_baostock_code,
    to_akshare_code,
    to_baostock_code,
)


class DataSyncTests(unittest.TestCase):
    def test_symbol_conversion(self) -> None:
        self.assertEqual(to_baostock_code("601838.SH"), "sh.601838")
        self.assertEqual(to_baostock_code("000001.SZ"), "sz.000001")
        self.assertEqual(from_baostock_code("sh.601838"), "601838.SH")
        self.assertEqual(to_akshare_code("601838.SH"), "601838")
        self.assertEqual(from_akshare_code("601838"), "601838.SH")
        self.assertEqual(from_akshare_code("000001"), "000001.SZ")
        self.assertEqual(from_akshare_code("920001"), "920001.BJ")

    def test_board_and_price_limit(self) -> None:
        self.assertEqual(_board_for("688001.SH"), "STAR")
        self.assertEqual(_board_for("300001.SZ"), "CHINEXT")
        self.assertEqual(_board_for("600000.SH"), "MAIN")
        self.assertEqual(_price_limit("688001.SH", "测试股份"), 0.20)
        self.assertEqual(_price_limit("600000.SH", "ST测试"), 0.05)

    def test_nan_values_become_database_null(self) -> None:
        records = _sanitize_records(
            [{"volume": float("nan"), "amount": math.inf, "close": 10.5}]
        )
        self.assertIsNone(records[0]["volume"])
        self.assertIsNone(records[0]["amount"])
        self.assertEqual(records[0]["close"], 10.5)

    def test_baostock_blacklist_is_fatal_login_error(self) -> None:
        self.assertTrue(_is_fatal_login_error("10001011"))
        message = _format_login_error("10001011", "黑名单用户，请与管理员联系")
        self.assertIn("继续重试无效", message)
        self.assertIn("recommend --as-of-date latest", message)
        self.assertFalse(_is_fatal_login_error("10001000"))

    def test_akshare_hist_normalization(self) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            [
                {
                    "日期": "2026-06-23",
                    "开盘": "10.1",
                    "最高": "10.5",
                    "最低": "10.0",
                    "收盘": "10.3",
                    "成交量": "1000",
                    "成交额": "10300",
                    "换手率": "1.2",
                    "涨跌额": "0.2",
                }
            ]
        )
        normalized = _normalize_akshare_hist(frame)
        self.assertEqual(normalized.loc[0, "date"], "2026-06-23")
        self.assertEqual(float(normalized.loc[0, "close"]), 10.3)


if __name__ == "__main__":
    unittest.main()
