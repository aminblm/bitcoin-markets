import unittest
from myopenproject import btc_analysis

class TestBTCAnalysis(unittest.TestCase):

    def test_average_price(self):
        prices = [10, 20, 30, 40, 50]
        expected_average = 30
        actual_average = btc_analysis.average_price(prices)
        self.assertEqual(expected_average, actual_average)
