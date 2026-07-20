# driftfilter

**A forward-only prototype filter for classifiers whose world moves after deployment.**

driftfilter is a compact, inspectable implementation inspired by [TMLR's 2026 STAD work on Bayesian filtering for temporal test-time adaptation.](https://openreview.net/forum?id=HFETOmUtrV).
It turns the paper's core idea into a deterministic benchmark that runs on a laptop with Python's standard library.

## Run it

```bash
python driftfilter.py
python -m unittest discover -s tests -v
```

The benchmark writes its result to stdout. Audio projects also write playable WAV files to `demo/`.

## What is tested

The test compares the research-inspired method with a deliberately legible baseline and requires
`accuracy_gain_pct >= 15`. The data generator is seeded, so the number in this README,
CI, and the portfolio case study can be reproduced.

## Scope

This is an educational research reproduction on controlled synthetic data. It is not a clinical,
diagnostic, production genomics, copyright-authentication, or safety-critical system. The point is
to make one mechanism measurable without hiding it behind a checkpoint or API.

## Research basis

- [TMLR's 2026 STAD work on Bayesian filtering for temporal test-time adaptation.](https://openreview.net/forum?id=HFETOmUtrV)
- Original implementation and benchmark in this repository are MIT licensed.

## License

MIT

## Reproduced result

| Metric | Value |
|---|---:|
| `frozen_accuracy` | **0.779** |
| `filtered_accuracy` | **1.0** |
| `accuracy_gain_pct` | **22.1** |
