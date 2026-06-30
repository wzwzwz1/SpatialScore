# Object Size Estimation Progress

## Current Status

`object_size_estimation` has a first deterministic 3D workflow in this repo.

The VSI-Bench task template is fixed:

```text
What is the length of the longest dimension (length, width, or height) of the <object>, measured in centimeters?
```

This means the workflow should estimate the object's longest 3D extent and return a numeric centimeter value. It does not need to infer a specific semantic axis such as height, width, or depth.

## Implemented Workflow

```text
video
-> sample frames
-> parse target object from question
-> LocalizeObjects(target)
-> select high-quality frames for size estimation
-> SAM2 mask from selected boxes
-> VGGT shared reconstruction over selected frames
-> extract target object mask point clouds
-> compute robust axis-aligned extent
-> aggregate per-frame longest extents
-> output size_centimeters
```

The first version is implemented as:

- tool: `spatial_agent.tools.object_size_3d.EstimateObjectSize3DTool`
- evaluator: `scripts/eval_vsibench_object_size_estimation.py`
- config key: `object_size_3d`

Default settings:

```text
top_size_frames: 6
size_aggregate: p90
extent_policy: p05_p95
mask_max_points: 256
min_mask_pixels: 1000
```

## Design Notes

The LLM should not estimate size directly. It should only route the task to the size tool and use the tool result.

The geometric target is:

```text
max_extent_cm = max(x_extent, y_extent, z_extent) * 100
```

where each axis extent is computed from robust point-cloud quantiles, currently `p05` to `p95`.

This differs from `object_abs_distance` only in the final measurement:

```text
object_abs_distance:
two object point clouds -> pairwise object distance -> meters

object_size_estimation:
one object point cloud -> longest robust extent -> centimeters
```

## Validation Completed

Local checks completed:

```text
python -m py_compile spatial_agent/tools/object_size_3d.py scripts/eval_vsibench_object_size_estimation.py
python scripts/eval_vsibench_object_size_estimation.py --limit 3 --list-only
pytest tests/spatial_agent -q
```

The test suite passed with `79 passed`.

## Next Work

1. Run a small baseline on 20 to 50 VSI-Bench samples.

   Goal: identify whether the first failure mode is localization, mask quality, VGGT scale, partial visibility, or aggregation.

2. Add verifier agents from `object_abs_distance`.

   Candidate path:

   ```text
   InstanceVerifierAgent
   -> Recheck
   -> BestMatchFallbackAgent
   -> FinalFrameVerifier
   ```

   For size estimation, verifier prompts should prefer boxes that show a complete or mostly complete target object, and should downgrade severe partial views instead of blindly accepting every category match.

3. Add instance selection for repeated same-class objects.

   If a room has multiple chairs, tables, or TVs, the current deterministic frame selection can mix views from multiple instances. A robust version should cluster candidate views by 3D center and select a dominant or most complete instance before computing size.

4. Tune extent and aggregation policies.

   Initial ablations:

   ```text
   extent_policy: p02_p98, p05_p95, p10_p90
   size_aggregate: p75, p90, max
   frame selection: top 4, top 6, top 8
   ```

5. Consider PCA/OBB extent after the axis-aligned baseline.

   Axis-aligned extent is simple and stable, but can overestimate rotated objects. PCA-oriented bounding-box extent may help after localization and instance selection are stable.

