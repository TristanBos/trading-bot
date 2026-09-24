import unittest
import pandas as pd
from research_engine import Config, Signal, _qty_for_risk, max_drawdown, summarize, resolve_intrabar_exit, next_bar_index

class TestEngine(unittest.TestCase):
    def test_risk_sizing(self):
        cfg = Config(risk_per_trade=0.01, max_position_value=2500)
        self.assertEqual(_qty_for_risk(10000, 100, 2, cfg), 25)

    def test_drawdown(self):
        s = pd.Series([100, 110, 99, 105], index=pd.date_range('2025-01-01', periods=4, freq='D'))
        self.assertAlmostEqual(max_drawdown(s), -0.1)

    def test_next_bar_entry(self):
        self.assertEqual(next_bar_index(4, 10), 5)
        self.assertIsNone(next_bar_index(9, 10))

    def test_stop_wins_collision(self):
        price, reason = resolve_intrabar_exit(1, 110, 90, 95, 105)
        self.assertEqual(reason, "stop")
        self.assertEqual(price, 95)

    def test_metrics(self):
        trades = pd.DataFrame([
            {"pnl":100.0,"side":"LONG","slippage_cost":1.0,"commission":0.0},
            {"pnl":-50.0,"side":"SHORT","slippage_cost":1.0,"commission":0.0},
        ])
        eq = pd.DataFrame({"equity":[10000,10050], "position":[0,10]}, index=pd.date_range('2025-01-01', periods=2, freq='D'))
        m = summarize(trades, eq, 10000)
        self.assertEqual(m['trades'],2)
        self.assertAlmostEqual(m['total_pnl'],50)

if __name__ == '__main__':
    unittest.main()
