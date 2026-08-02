import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import driftfilter
import driftfilter_v2
from adversarial import HOLDOUT_SEEDS, STRESS_SHIFT, TUNING_SEEDS
from eval_v2 import summarize


class AdversarialTest(unittest.TestCase):
    def test_holdout_disjoint_from_tuning(self):
        self.assertTrue(set(TUNING_SEEDS).isdisjoint(HOLDOUT_SEEDS))

    def test_original_benchmark_still_reproduces_exactly(self):
        result = driftfilter.run()
        self.assertEqual(result["frozen_accuracy"], 0.779)
        self.assertEqual(result["filtered_accuracy"], 1.0)
        self.assertEqual(result["accuracy_gain_pct"], 22.1)

    def test_v2_position_only_baseline_matches_original_module_at_published_rate(self):
        result = driftfilter_v2.run_position_only(seed=19, total_shift=3.4)
        self.assertEqual(result, driftfilter.run(seed=19))

    def test_original_bug_plain_ema_degrades_under_faster_drift(self):
        """driftfilter.py's filtered_accuracy hits a seed-invariant 1.0 at
        the published drift rate, but the plain position-only EMA falls
        behind at faster (still linear, still plausible) drift -- degrading
        to well below 1.0, sometimes closer to chance."""
        accs = [
            driftfilter_v2.run_position_only(seed=seed, total_shift=STRESS_SHIFT)["filtered_accuracy"]
            for seed in TUNING_SEEDS
        ]
        self.assertLess(sum(accs) / len(accs), 0.75)

    def test_v2_fix_generalizes_on_tuning_seeds(self):
        result = summarize(TUNING_SEEDS)
        self.assertGreater(result["v2_mean_filtered_acc"], result["original_mean_filtered_acc"])
        self.assertGreater(result["v2_min_filtered_acc"], 0.95)

    def test_v2_fix_generalizes_on_frozen_holdout_seeds(self):
        result = summarize(HOLDOUT_SEEDS)
        self.assertGreater(result["v2_mean_filtered_acc"], result["original_mean_filtered_acc"])
        self.assertGreater(result["v2_min_filtered_acc"], 0.95)

    def test_v2_does_not_regress_the_original_published_seed(self):
        result = driftfilter_v2.run(seed=19, total_shift=3.4)
        self.assertEqual(result["accuracy_gain_pct"], 22.1)

    def test_original_module_untouched(self):
        import inspect

        source = inspect.getsource(driftfilter.run)
        self.assertIn("rate=.08", source)
        self.assertNotIn("velocity", source)

    def test_report_is_reproducible(self):
        a = summarize(TUNING_SEEDS[:5])
        b = summarize(TUNING_SEEDS[:5])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
