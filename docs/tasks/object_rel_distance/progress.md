# Object Rel Distance Progress

## Current Status

We now have a working `object_rel_distance` workflow in SpatialAgent:

1. Sample 64 frames from the video.
2. Localize the reference object and candidate objects on shared frames.
3. Select shared frames for VGGT reconstruction.
4. Use SAM2 masks to lift detections into 3D point clouds.
5. For the reference object, keep only the highest-confidence valid instance.
6. Compare the reference point cloud against each candidate point cloud.

## What Improved

- Reference-object false positives were a major source of error.
- Switching the reference object to a single highest-confidence valid instance fixed cases like `doc_523`.
- On a 20-sample `object_rel_distance` run, the current pipeline reached:
  - `success`: 17/20
  - `correct`: 15/20

## Remaining Issues

- Some samples still fail because the reference object has no finite point cloud.
- Some samples still fail because localization cannot find a usable shared frame.
- Candidate ranking is better than before, but not fully stable.

## Notes

- The current implementation is empirical and sample-driven.
- The main debugging signal is the structured `result.json`, not the raw overlay image path alone.
- The latest 200-sample baseline analysis is recorded in `random200_analysis_2026-05-28.md`.
