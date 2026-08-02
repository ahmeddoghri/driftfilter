"""Compare position-only EMA (driftfilter.py's method) to position+velocity
tracking (driftfilter_v2.py) under drift roughly 4.4x faster than the
published rate, where the plain EMA falls behind."""
import json
import statistics as st

from adversarial import HOLDOUT_SEEDS, STRESS_SHIFT, TUNING_SEEDS
from driftfilter_v2 import run, run_position_only


def summarize(seeds):
    orig = [run_position_only(seed=seed, total_shift=STRESS_SHIFT)["filtered_accuracy"] for seed in seeds]
    v2 = [run(seed=seed, total_shift=STRESS_SHIFT)["filtered_accuracy"] for seed in seeds]
    return {
        "n": len(seeds),
        "stress_shift": STRESS_SHIFT,
        "original_mean_filtered_acc": round(st.mean(orig), 3),
        "original_min_filtered_acc": min(orig),
        "v2_mean_filtered_acc": round(st.mean(v2), 3),
        "v2_min_filtered_acc": min(v2),
    }


def main():
    print("driftfilter eval_v2: position-only EMA vs. position+velocity tracking under faster drift")
    for label, seeds in (("tuning", TUNING_SEEDS), ("holdout", HOLDOUT_SEEDS)):
        print(f"\n{label} ({len(seeds)} seeds):")
        print(json.dumps(summarize(seeds), indent=2))


if __name__ == "__main__":
    main()
