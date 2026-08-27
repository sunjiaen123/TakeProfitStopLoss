import unittest
import math
from unittest.mock import patch

from tpsl.data_sync import (
    _board_for,
    _download_symbol_akshare,
    _format_login_error,
    _looks_like_etf_symbol,
    _normalize_akshare_hist,
    _is_fatal_login_error,
    _sanitize_records,
    _price_limit,
    _position_universe,
    _security_details_for_symbol,
    _supported_baostock_types,
    fetch_universe,
    from_akshare_code,
    from_akshare_etf_code,
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

    def test_etf_symbol_conversion_and_detection(self) -> None:
        self.assertEqual(from_akshare_etf_code("513770"), "513770.SH")
        self.assertEqual(from_akshare_etf_code("159915"), "159915.SZ")
        self.assertTrue(_looks_like_etf_symbol("513770.SH"))
        self.assertTrue(_looks_like_etf_symbol("159915.SZ"))
        self.assertFalse(_looks_like_etf_symbol("601838.SH"))
        with self.assertRaises(ValueError):
            from_akshare_etf_code("601838")

    def test_position_symbols_validate_exchange_suffix(self) -> None:
        self.assertEqual(
            _security_details_for_symbol("603993.SH"),
            ("stock", "SH"),
        )
        self.assertEqual(
            _security_details_for_symbol("002636.SZ"),
            ("stock", "SZ"),
        )
        with self.assertRaisesRegex(ValueError, "603993.SH"):
            _security_details_for_symbol("603993.SZ")

    def test_position_universe_does_not_need_remote_market_list(self) -> None:
        universe = _position_universe(
            object(),
            ["002636.SZ", "603993.SH", "513770.SH"],
        )
        self.assertEqual(
            universe["symbol"].tolist(),
            ["002636.SZ", "603993.SH", "513770.SH"],
        )
        self.assertEqual(
            universe["security_type"].tolist(),
            ["stock", "stock", "etf"],
        )

    def test_baostock_etfs_are_opt_in(self) -> None:
        self.assertEqual(_supported_baostock_types(False), {"1"})
        self.assertEqual(_supported_baostock_types(True), {"1", "5"})

    def test_auto_stock_source_falls_back_on_baostock_network_error(self) -> None:
        import pandas as pd
        from datetime import date

        fallback = pd.DataFrame(
            [{"symbol": "601838.SH", "security_type": "stock"}]
        )
        with patch(
            "tpsl.data_sync.fetch_baostock_universe",
            side_effect=RuntimeError("BaoStock 10002007: 网络接收错误"),
        ), patch(
            "tpsl.data_sync.fetch_akshare_universe",
            return_value=fallback,
        ) as akshare_fetch:
            universe, source = fetch_universe(
                "auto",
                date(2026, 8, 12),
                retry_count=1,
                retry_delay=0.0,
                socket_timeout=1.0,
                include_etfs=False,
            )

        self.assertEqual(source, "akshare")
        self.assertEqual(universe.loc[0, "symbol"], "601838.SH")
        akshare_fetch.assert_called_once_with(date(2026, 8, 12), False)

    def test_auto_etf_source_prefers_akshare(self) -> None:
        import pandas as pd
        from datetime import date

        universe_frame = pd.DataFrame(
            [{"symbol": "513770.SH", "security_type": "etf"}]
        )
        with patch(
            "tpsl.data_sync.fetch_akshare_universe",
            return_value=universe_frame,
        ) as akshare_fetch, patch(
            "tpsl.data_sync.fetch_baostock_universe"
        ) as baostock_fetch:
            universe, source = fetch_universe(
                "auto",
                date(2026, 8, 16),
                retry_count=1,
                retry_delay=0.0,
                socket_timeout=1.0,
                include_etfs=True,
            )

        self.assertEqual(source, "akshare")
        self.assertEqual(universe.loc[0, "symbol"], "513770.SH")
        akshare_fetch.assert_called_once_with(date(2026, 8, 16), True)
        baostock_fetch.assert_not_called()

    def test_auto_etf_source_falls_back_to_baostock(self) -> None:
        import pandas as pd
        from datetime import date

        fallback = pd.DataFrame(
            [{"symbol": "513770.SH", "security_type": "etf"}]
        )
        with patch(
            "tpsl.data_sync.fetch_akshare_universe",
            side_effect=RuntimeError("ETF 列表不可用"),
        ), patch(
            "tpsl.data_sync.fetch_baostock_universe",
            return_value=fallback,
        ) as baostock_fetch:
            universe, source = fetch_universe(
                "auto",
                date(2026, 8, 16),
                retry_count=1,
                retry_delay=0.0,
                socket_timeout=1.0,
                include_etfs=True,
            )

        self.assertEqual(source, "baostock")
        self.assertEqual(universe.loc[0, "symbol"], "513770.SH")
        baostock_fetch.assert_called_once_with(
            date(2026, 8, 16), 1, 0.0, 1.0, True
        )

    def test_explicit_baostock_source_does_not_fall_back(self) -> None:
        from datetime import date

        with patch(
            "tpsl.data_sync.fetch_baostock_universe",
            side_effect=RuntimeError("BaoStock 10002007: 网络接收错误"),
        ), patch("tpsl.data_sync.fetch_akshare_universe") as akshare_fetch:
            with self.assertRaisesRegex(RuntimeError, "10002007"):
                fetch_universe(
                    "baostock",
                    date(2026, 8, 12),
                    retry_count=1,
                    retry_delay=0.0,
                    socket_timeout=1.0,
                    include_etfs=True,
                )

        akshare_fetch.assert_not_called()

    def test_auto_source_reports_both_failures(self) -> None:
        from datetime import date

        with patch(
            "tpsl.data_sync.fetch_baostock_universe",
            side_effect=TimeoutError("timed out"),
        ), patch(
            "tpsl.data_sync.fetch_akshare_universe",
            side_effect=RuntimeError("ETF 列表不可用"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "AkShare 与 BaoStock 均不可用",
            ) as caught:
                fetch_universe(
                    "auto",
                    date(2026, 8, 12),
                    retry_count=1,
                    retry_delay=0.0,
                    socket_timeout=1.0,
                    include_etfs=True,
                )

        self.assertIn("timed out", str(caught.exception))
        self.assertIn("ETF 列表不可用", str(caught.exception))

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

    def test_akshare_etf_download_uses_fund_endpoint(self) -> None:
        import pandas as pd

        history = pd.DataFrame(
            [
                {
                    "日期": "2026-08-07",
                    "开盘": "1.001",
                    "最高": "1.015",
                    "最低": "0.998",
                    "收盘": "1.010",
                    "成交量": "100000",
                    "成交额": "101000",
                    "换手率": "2.1",
                    "涨跌额": "0.010",
                }
            ]
        )
        with patch(
            "akshare.fund_etf_hist_em",
            return_value=history,
        ) as endpoint:
            result = _download_symbol_akshare(
                (
                    "513770.SH",
                    "2026-08-07",
                    "2026-08-07",
                    "ETF",
                    "etf",
                    1,
                    0.0,
                    1.0,
                )
            )

        self.assertEqual(result["error"], "")
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["symbol"], "513770.SH")
        self.assertEqual(result["rows"][0]["industry_code"], "ETF")
        self.assertEqual(endpoint.call_count, 2)

    def test_akshare_etf_history_falls_back_to_sina(self) -> None:
        import pandas as pd

        sina_history = pd.DataFrame(
            [
                {
                    "date": "2026-08-14",
                    "open": "0.363",
                    "high": "0.365",
                    "low": "0.359",
                    "close": "0.361",
                    "volume": "1183707599",
                    "amount": "428334704",
                }
            ]
        )
        with patch(
            "akshare.fund_etf_hist_em",
            side_effect=ConnectionError("remote closed"),
        ), patch(
            "akshare.fund_etf_hist_sina",
            return_value=sina_history,
        ) as sina_endpoint:
            result = _download_symbol_akshare(
                (
                    "513770.SH",
                    "2026-08-01",
                    "2026-08-16",
                    "ETF",
                    "etf",
                    1,
                    0.0,
                    1.0,
                )
            )

        self.assertEqual(result["error"], "")
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["trade_date"], "2026-08-14")
        self.assertEqual(result["rows"][0]["source"], "AKSHARE_SINA")
        sina_endpoint.assert_called_once_with(symbol="sh513770")

    def test_akshare_stock_history_falls_back_to_sina(self) -> None:
        import pandas as pd

        sina_history = pd.DataFrame(
            [
                {
                    "date": "2026-08-21",
                    "open": "405.0",
                    "high": "411.0",
                    "low": "400.1",
                    "close": "409.06",
                    "volume": "34076661",
                    "amount": "13868010000",
                    "turnover": "0.050975",
                }
            ]
        )
        with patch(
            "akshare.stock_zh_a_hist",
            side_effect=ConnectionError("eastmoney unavailable"),
        ), patch(
            "akshare.stock_zh_a_daily",
            return_value=sina_history,
        ) as sina_endpoint:
            result = _download_symbol_akshare(
                (
                    "603986.SH",
                    "2026-08-21",
                    "2026-08-21",
                    "UNKNOWN",
                    "stock",
                    1,
                    0.0,
                    1.0,
                )
            )

        self.assertEqual(result["error"], "")
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["trade_date"], "2026-08-21")
        self.assertEqual(result["rows"][0]["source"], "AKSHARE_SINA")
        self.assertAlmostEqual(result["rows"][0]["turnover_rate"], 0.050975)
        self.assertEqual(sina_endpoint.call_count, 2)


if __name__ == "__main__":
    unittest.main()
