# object_counting 结果记录

整理日期：2026-06-30

## 当前完整评测结果

| 项目 | 数值 |
| --- | ---: |
| 评测集 | VSI-Bench test |
| 总样本数 | 565 |
| 可评测样本数 | 565 |
| 当前最佳阈值 | 0.8 |
| MRA | 44.37% |
| Exact accuracy | 29.38% |
| MAE | 1.515 |

## 当前工作流

```text
视频采样
+ Rex-Omni 逐帧目标检测
+ VGGT 三维点云提升
+ 3D constrained greedy clustering
+ 阈值扫描得到最终计数
```

当前 canonical run 扫描了多个 3D 聚类距离阈值，最佳为 `0.8`。

## 阈值扫描结果

| threshold | MRA | Exact accuracy | MAE |
| ---: | ---: | ---: | ---: |
| 0.35 | 7.29% | 2.12% | 6.165 |
| 0.40 | 11.59% | 4.60% | 4.899 |
| 0.45 | 15.68% | 6.19% | 3.846 |
| 0.50 | 21.36% | 10.80% | 3.147 |
| 0.55 | 27.65% | 16.46% | 2.625 |
| 0.60 | 31.65% | 19.29% | 2.211 |
| 0.65 | 36.51% | 23.01% | 1.933 |
| 0.70 | 40.32% | 25.49% | 1.731 |
| 0.75 | 42.12% | 27.43% | 1.618 |
| 0.80 | 44.37% | 29.38% | 1.515 |

## 结果文件

```text
run dir:
/disk/wangzhe/SpatialScore/runs/object_counting_full_20260618_gpu3

records:
/disk/wangzhe/SpatialScore/runs/object_counting_full_20260618_gpu3/records.json

threshold scan:
/disk/wangzhe/SpatialScore/runs/object_counting_full_20260618_gpu3/threshold_scan.json
```

## 备注

日志中出现过 HuggingFace/proxy retry warning，但最终完整写出了 `records.json` 和 `threshold_scan.json`，本次结果已落盘。

