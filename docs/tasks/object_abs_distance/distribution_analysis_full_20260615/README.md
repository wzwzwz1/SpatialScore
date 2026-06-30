# Object Absolute Distance Distribution Analysis

This report quantifies which point-cloud distance distribution statistic works best per answer.

- Samples: `771`
- Per-answer oracle MRA: `48.61%`

## Fixed Aggregates

| aggregate | MRA | MAE |
| --- | ---: | ---: |
| `min` | 3.10% | 1.990 |
| `p05` | 7.06% | 1.818 |
| `p10` | 8.78% | 1.753 |
| `p25` | 14.62% | 1.590 |
| `median` | 22.67% | 1.356 |
| `p75` | 29.66% | 1.182 |
| `p90` | 31.56% | 1.141 |

## Best Aggregate Per Answer

`best_tiebreak_counts` uses VSI-Bench MRA first and absolute error as the tie-breaker.

```json
{
  "median": 67,
  "p75": 77,
  "p90": 550,
  "p05": 15,
  "p10": 13,
  "p25": 38,
  "min": 11
}
```

## Interpretation

- The best statistic is not fixed across samples.
- `median` is the strongest single fixed statistic on the current 40-sample run.
- Low quantiles help on some near/contact cases, but are harmful if used globally.
- This supports an adaptive distribution-selection module as a separable research contribution.
