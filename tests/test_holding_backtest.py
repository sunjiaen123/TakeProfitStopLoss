import unittest

from tpsl.holding_backtest import _maximum_drawdown, ratchet_stop


class HoldingBacktestTests(unittest.TestCase):
    def test_stop_only_moves_up(self) -> None:
        self.assertEqual(ratchet_stop(None, 9.5), 9.5)
        self.assertEqual(ratchet_stop(9.5, 9.2), 9.5)
        self.assertEqual(ratchet_stop(9.5, 9.8), 9.8)

    def test_maximum_drawdown(self) -> None:
        drawdown = _maximum_drawdown([1.0, 1.1, 1.05, 0.99, 1.2])
        self.assertAlmostEqual(drawdown, 0.99 / 1.1 - 1)


if __name__ == "__main__":
    unittest.main()
