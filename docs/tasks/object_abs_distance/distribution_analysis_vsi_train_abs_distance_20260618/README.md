# Object Absolute Distance Distribution Analysis

This report quantifies which point-cloud distance distribution statistic works best per answer.

- Samples: `1311`
- Per-answer oracle MRA: `44.72%`

## Fixed Aggregates

| aggregate | MRA | MAE |
| --- | ---: | ---: |
| `min` | 2.43% | 2.074 |
| `p05` | 6.03% | 1.903 |
| `p10` | 7.82% | 1.838 |
| `p25` | 12.50% | 1.679 |
| `median` | 19.12% | 1.442 |
| `p75` | 26.99% | 1.259 |
| `p90` | 30.12% | 1.176 |

## Best Aggregate Per Answer

`best_tiebreak_counts` uses VSI-Bench MRA first and absolute error as the tie-breaker.

```json
{
  "p90": 987,
  "p75": 116,
  "p25": 66,
  "median": 89,
  "min": 13,
  "p10": 23,
  "p05": 17
}
```

## Interpretation

- The best statistic is not fixed across samples.
- `median` is the strongest single fixed statistic on the current 40-sample run.
- Low quantiles help on some near/contact cases, but are harmful if used globally.
- This supports an adaptive distribution-selection module as a separable research contribution.
