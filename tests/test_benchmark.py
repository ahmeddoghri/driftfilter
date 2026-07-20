import unittest
import driftfilter

class BenchmarkTest(unittest.TestCase):
    def test_research_method_clears_baseline(self):
        result = driftfilter.run()
        self.assertGreaterEqual(result["accuracy_gain_pct"], 15)

if __name__ == "__main__":
    unittest.main()
