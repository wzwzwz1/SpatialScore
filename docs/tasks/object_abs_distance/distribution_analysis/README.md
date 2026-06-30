# Object Absolute Distance Distribution Analysis

This report quantifies which point-cloud distance distribution statistic works best per answer.

- Samples: `40`
- Per-answer oracle MRA: `44.50%`

## Fixed Aggregates

| aggregate | MRA | MAE |
| --- | ---: | ---: |
| `min` | 2.50% | 2.364 |
| `p05` | 10.50% | 2.192 |
| `p10` | 11.25% | 2.131 |
| `p25` | 14.50% | 1.975 |
| `median` | 24.75% | 1.752 |
| `p75` | 22.25% | 1.586 |
| `p90` | 21.50% | 1.601 |

## Best Aggregate Per Answer

`best_tiebreak_counts` uses VSI-Bench MRA first and absolute error as the tie-breaker.

```json
{
  "median": 8,
  "p75": 5,
  "p90": 24,
  "p05": 1,
  "p10": 1,
  "p25": 1
}
```

## Interpretation

- The best statistic is not fixed across samples.
- `median` is the strongest single fixed statistic on the current 40-sample run.
- Low quantiles help on some near/contact cases, but are harmful if used globally.
- This supports an adaptive distribution-selection module as a separable research contribution.
