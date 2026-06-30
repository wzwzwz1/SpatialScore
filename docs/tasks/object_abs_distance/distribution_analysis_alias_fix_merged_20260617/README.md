# Object Absolute Distance Distribution Analysis

This report quantifies which point-cloud distance distribution statistic works best per answer.

- Samples: `818`
- Per-answer oracle MRA: `48.81%`

## Fixed Aggregates

| aggregate | MRA | MAE |
| --- | ---: | ---: |
| `min` | 3.40% | 1.968 |
| `p05` | 7.51% | 1.797 |
| `p10` | 9.32% | 1.734 |
| `p25` | 15.06% | 1.571 |
| `median` | 22.86% | 1.344 |
| `p75` | 29.71% | 1.176 |
| `p90` | 31.76% | 1.133 |

## Best Aggregate Per Answer

`best_tiebreak_counts` uses VSI-Bench MRA first and absolute error as the tie-breaker.

```json
{
  "median": 69,
  "p75": 79,
  "p90": 584,
  "p05": 16,
  "p10": 15,
  "p25": 43,
  "min": 12
}
```

## Interpretation

- The best statistic is not fixed across samples.
- `median` is the strongest single fixed statistic on the current 40-sample run.
- Low quantiles help on some near/contact cases, but are harmful if used globally.
- This supports an adaptive distribution-selection module as a separable research contribution.
