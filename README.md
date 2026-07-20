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

## Research basis

- [TMLR's 2026 STAD work on Bayesian filtering for temporal test-time adaptation](https://openreview.net/forum?id=HFETOmUtrV)
- Original implementation and benchmark in this repository are MIT licensed.

## License

MIT
