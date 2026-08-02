# driftfilter

**A forward-only prototype filter for classifiers whose world moves after deployment.**

A classifier frozen at training time is a GPS that refuses to reroute. The world drifts — user behavior shifts, sensor calibration wanders, class centroids migrate across the feature space — and a frozen model keeps confidently pointing at where things used to be. driftfilter doesn't retrain, doesn't need new labels, and doesn't stop and ask for directions. It just nudges its class prototypes toward wherever the incoming data actually is, one point at a time, forward-only.

It's a compact, inspectable implementation inspired by [TMLR's 2026 STAD work on Bayesian filtering for temporal test-time adaptation](https://openreview.net/forum?id=HFETOmUtrV), rebuilt small enough to read in one sitting and run without a GPU, a checkpoint, or an API key.

## The result

```bash
python driftfilter.py
```
```json
{
  "frozen_accuracy": 0.779,
  "filtered_accuracy": 1.0,
  "accuracy_gain_pct": 22.1
}
```

Classify 240 sequential points against class centroids that never move and you get 77.9% accuracy — solid until the drift catches up to you, then steadily less so. Let the centroids filter toward each new observation as it arrives, using nothing but the point itself, and accuracy holds at 100% the entire way through — a 22.1 percentage-point gain earned with zero labels and zero retraining.

**Update:** that perfect 100% only holds at the one drift rate this
benchmark ships with. The filter tracks centroid *position* alone with a
plain EMA; push the same linear drift about 4.4x faster (still smooth,
still plausible) and it falls behind, degrading to a mean 68.6% accuracy
across dozens of seeds — no longer meaningfully better than the frozen
baseline. `driftfilter_v2.py` also tracks centroid *velocity* (an actual
Kalman-style filter, closer to what the cited research does) and holds
~100% accuracy at that same faster rate. Details below.

## How it works

Two class centroids start in the right place, then drift linearly over 240 timesteps while wobbling with a sine term, simulating a slow, real, structured shift rather than random noise. The frozen baseline classifies every point by nearest centroid and never updates. The filtered version does the same nearest-centroid classification, but after each prediction nudges the *predicted* centroid a small step toward the observed point — an exponential moving average acting as a crude Kalman-style filter. No label feedback, no gradient step, just "move toward what you just saw."

## Run it

```bash
python driftfilter.py
python -m unittest discover -s tests -v
```

## What is tested

The test compares the filtered classifier against the frozen baseline and requires `accuracy_gain_pct >= 15`. The data generator is seeded, so the number in this README, in CI, and in the portfolio case study are the same number, not three different ones that happen to rhyme.

## Scope

This is an educational research reproduction on a controlled synthetic drift trajectory. It is not a clinical, diagnostic, production ML monitoring, or safety-critical system, and it makes no claim about real-world drift datasets. The point is to make one mechanism — unsupervised prototype filtering beats a frozen classifier under drift — measurable without hiding it behind a checkpoint.

## Position-only tracking only works at one drift speed

`driftfilter.py`'s filter nudges the predicted centroid's *position* a
fixed 8% toward each new point — a plain exponential moving average, with
no notion of how fast the centroid is actually moving. At the published
drift rate (`total_shift=3.4` over 240 steps) that's plenty fast to keep
up, and `filtered_accuracy` hits a seed-invariant `1.0`. It isn't a
tautology — the classifier genuinely has to get every single point right —
but it's also a claim characterized at exactly one point on a curve.

```bash
python eval_v2.py
```
```
tuning (40 seeds):  original_mean=0.686  original_min=0.629  v2_mean=1.000  v2_min=0.996
holdout (30 seeds): original_mean=0.686  original_min=0.617  v2_mean=1.000  v2_min=1.000
```

Push the drift about 4.4x faster (`total_shift=15`, still linear, still an
equally plausible real-world drift rate) and the plain-EMA filter falls
behind: mean `filtered_accuracy` drops to 68.6% across 40 tuning seeds and
a disjoint 30-seed holdout (evaluated once), min as low as 61.7% — barely
better than the frozen baseline it's supposed to beat by 22 points.
`driftfilter_v2.py` adds a velocity state: it predicts each centroid's
position forward before comparing, then updates both position and velocity
from the observed displacement, the same idea a real Kalman filter uses.
At that same 4.4x-faster drift rate it holds ~100% mean accuracy (min 99.6%
tuning / 100% holdout) on both sweeps, and it reproduces the original
22.1pp number exactly at the published drift rate. `driftfilter.py` is
untouched. (Checked separately: for classes drifting *toward* each other
instead of together, neither method offers much advantage over frozen,
since the frozen decision boundary already handles that symmetric case
reasonably well — an honest limitation, not something papered over here.)

## Research basis

- [TMLR's 2026 STAD work on Bayesian filtering for temporal test-time adaptation](https://openreview.net/forum?id=HFETOmUtrV)
- Original implementation and benchmark in this repository are MIT licensed.

## License

MIT
