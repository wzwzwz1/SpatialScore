# object_rel_distance 结果记录

整理日期：2026-06-30

## 当前完整评测结果

| 项目 | 数值 |
| --- | ---: |
| 评测集 | VSI-Bench test |
| 总样本数 | 710 |
| 成功 / 可评测样本数 | 610 |
| 正确数 | 264 |
| Success-only accuracy | 43.28% |
| Full accuracy | 37.18% |
| 未成功 / 不可评测 | 100 |

## 当前工作流

```text
视频采样
+ Rex / GroundingDINO 目标检测
+ SAM2 掩码
+ VGGT 三维重建
+ 参考物与候选物体点云距离排序
+ bbox contact shortcut + 3D 距离校验
```

当前 canonical run 使用完整 `object_rel_distance` test 样本，按选项预测最近 / 最远目标，指标为选项准确率。

## 结果文件

```text
run dir:
/disk/wangzhe/SpatialScore/runs/object_rel_distance_full_20260618_gpu2

summary:
/disk/wangzhe/SpatialScore/runs/object_rel_distance_full_20260618_gpu2/summary.json

records:
/disk/wangzhe/SpatialScore/runs/object_rel_distance_full_20260618_gpu2/records.json
```

## summary.json

```json
{
  "count": 710,
  "evaluated": 610,
  "success": 610,
  "correct": 264
}
```

