import unittest

import numpy as np
import pandas as pd

from tpsl.features import FEATURE_COLUMNS, build_feature_frame


class FeatureTests(unittest.TestCase):
    def test_build_features_and_next_day_labels(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=100)
        rows = []
        for symbol, industry, offset in [
            ("000001.SZ", "BANK", 0.0),
            ("600000.SH", "BANK", 2.0),
            ("300001.SZ", "TECH", 5.0),
        ]:
            for index, trade_date in enumerate(dates):
                close = 10 + offset + index * 0.03 + np.sin(index / 5) * 0.1
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "open": close - 0.02,
                        "high": close + 0.10,
                        "low": close - 0.12,
                        "close": close,
                        "volume": 1_000_000 + index * 1000,
                        "amount": close * (1_000_000 + index * 1000),
                        "industry": industry,
                    }
                )
        frame = build_feature_frame(pd.DataFrame(rows), workers=3, min_symbol_rows=80)
        usable = frame.dropna(subset=FEATURE_COLUMNS)
        self.assertGreater(len(usable), 150)
        first_symbol = frame[frame["symbol"] == "000001.SZ"].reset_index(drop=True)
        expected = first_symbol.loc[1, "high"] / first_symbol.loc[0, "close"] - 1
        self.assertAlmostEqual(first_symbol.loc[0, "next_high_return"], expected)


if __name__ == "__main__":
    unittest.main()

