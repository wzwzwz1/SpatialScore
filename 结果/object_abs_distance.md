# object_abs_distance 结果记录

整理日期：2026-06-30

## 当前最佳结果

| 项目 | 数值 |
| --- | ---: |
| 评测集 | VSI-Bench test |
| 总样本数 | 834 |
| 成功 / 可评测样本数 | 818 |
| Success-only MRA | 33.24% |
| Full MRA | 约 32.60% |

## 当前最佳工作流

```text
alias expansion
+ 多 agent 检测复核 / 实例级拒识 / best-match fallback
+ VGGT 三维重建
+ DADP 分布自适应修正
```

DADP 当前采用 `p90 -> p75` 的质量感知修正：

```text
动作空间：{p75, p90}
模型：DecisionTreeRegressor max_depth=3
训练目标：delta = mra_p75 - mra_p90
训练权重：max(abs(delta), 0.05) * cleanliness^2
默认分位数：p90
修正逻辑：当模型预测切换有收益时，从 p90 切到 p75
```

## 关键对比

| 方法 | 总数 | 成功 | Success-only MRA | Full MRA |
| --- | ---: | ---: | ---: | ---: |
| original full run, fixed `p90` | 834 | 771 | 31.56% | 29.17% |
| alias-fix merged, fixed `p90` | 834 | 818 | 31.76% | 31.15% |
| train-learned `core2` gate | 834 | 818 | 32.79% | 约 32.16% |
| cleanliness-weighted DADP | 834 | 818 | 33.24% | 约 32.60% |

## 结果路径

```text
/disk/wangzhe/SpatialScore/runs/object_abs_distance_full_qwen25vl_gpu3_20260615_160000
/disk/wangzhe/SpatialScore/runs/object_abs_distance_full_alias_fix_merged_20260617
/disk/wangzhe/SpatialScore/docs/tasks/object_abs_distance/dadp_cleanliness_weight_20260618
/disk/wangzhe/discovery/dadp_status.md
```

