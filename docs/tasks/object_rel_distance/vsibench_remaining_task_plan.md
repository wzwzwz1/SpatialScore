# VSI-Bench Remaining Task Plan

This plan tracks how to reuse the current object counting and 3D object distance
workflows for the remaining VSI-Bench task types.

## Existing Implemented Workflows

### Object Counting

Tool:

- `CountVideoObjects3D`

Workflow:

- sample video frames,
- detect/count object views,
- lift views to VGGT 3D points,
- cluster repeated views of the same instance.

Primary target:

- `object_counting`

### Object Absolute Distance

Tool:

- `EstimateObjectDistance3D`

Workflow:

- sample video frames,
- localize two objects across frames,
- select top multi-object frames,
- get SAM2 object masks,
- run VGGT multi-frame reconstruction,
- compute object-to-object mask point-cloud distance.

Primary target:

- `object_abs_distance`

Current best dev setting:

- `top_distance_frames=6`
- `pointcloud_aggregate=p90`
- `mask_max_points=128`

## Remaining Task Types

VSI-Bench test split task counts:

| Task type | Count | Reuse level |
| --- | ---: | --- |
| `object_rel_distance` | 710 | High |
| `object_size_estimation` | 953 | High |
| `obj_appearance_order` | 618 | Medium |
| `object_rel_direction_easy` | 217 | Medium |
| `object_rel_direction_medium` | 378 | Medium |
| `object_rel_direction_hard` | 373 | Medium |
| `room_size_estimation` | 288 | Medium/Low |
| `route_planning` | 194 | Low |

## Priority 1: `object_rel_distance`

Goal:

- Answer relative distance questions by comparing computed 3D object distances.

Reuse:

- `EstimateObjectDistance3D`
- multi-frame localization
- SAM2 masks
- VGGT object point-cloud distance

Expected implementation:

- Add a lightweight reasoning/answering layer that:
  - parses the reference object and candidate objects from question/options,
  - calls object distance estimation for each candidate pair,
  - chooses the smaller/larger distance according to the question.

Potential tool:

- `CompareObjectDistance3D`

Inputs:

- `images`
- `reference_object`
- `candidate_objects`
- `mode`: `nearest` or `farthest`

Output:

- selected object/option,
- per-candidate distances,
- selected frame positions and distance stats for each candidate.

Validation:

- run first 20 `object_rel_distance` samples,
- report exact match with ground truth/options,
- inspect failures for parsing vs geometry errors.

## Priority 2: `object_size_estimation`

Goal:

- Estimate a physical object size from its 3D mask point cloud.

Reuse:

- multi-frame localization,
- SAM2 masks,
- VGGT world points,
- point-cloud filtering.

Expected implementation:

- Add `EstimateObjectSize3D`.
- Select top frames where the target object is clear.
- Extract object mask point clouds.
- Estimate size using robust point-cloud extents:
  - p05-p95 extent per axis,
  - max robust extent,
  - optional oriented extent later.

Risk:

- VGGT scale and partial views may bias size.
- Need to understand VSI-Bench answer format before final scoring.

## Priority 3: `obj_appearance_order`

Goal:

- Determine which object appears first in the video.

Reuse:

- frame sampling,
- multi-frame object localization.

Expected implementation:

- Add `EstimateObjectAppearanceOrder`.
- Run localization for candidate objects across sampled frames.
- Use first stable detection frame, not a single noisy detection.

Validation:

- first 20 `obj_appearance_order` samples.

## Priority 4: `object_rel_direction_*`

Goal:

- Answer relative direction questions.

Reuse:

- localization,
- VGGT object centers,
- camera parameters.

Expected implementation:

- Start with `easy` using 2D bbox/visual frame rules.
- For `medium/hard`, estimate 3D centers and compare in an explicit coordinate frame.

Open issue:

- Need to define whether the question expects camera-view direction, trajectory direction,
  or scene/world direction.

## Lower Priority: `room_size_estimation` And `route_planning`

These require scene-level geometry rather than object-level geometry.

Possible reuse:

- VGGT multi-frame reconstruction,
- camera poses,
- scene point cloud.

Likely additional work:

- floor/wall/free-space estimation,
- path planning graph,
- robust room extent estimation.

## Immediate Next Step

Implement `CompareObjectDistance3D` for `object_rel_distance`, using
`EstimateObjectDistance3D` as the geometric primitive.

Initial scope:

- support option-style questions where options are candidate object names,
- support nearest/farthest wording,
- run on 20 samples,
- record success rate and failure modes.

## Progress: `object_rel_distance`

Implemented:

- `CompareObjectDistance3D`
- agent prompt preference for relative distance questions
- runtime image injection for `CompareObjectDistance3D`
- evaluation script: `scripts/eval_vsibench_object_rel_distance.py`

Initial smoke test on first 3 `object_rel_distance` samples:

| Samples | Correct | Notes |
| ---: | ---: | --- |
| 3 | 2 | all tool calls succeeded |

Naive pairwise 20-sample result:

| Strategy | Samples | Success | Correct |
| --- | ---: | ---: | ---: |
| pairwise `EstimateObjectDistance3D` per candidate | 20 | 20 | 3 |

This showed that pairwise absolute distances are not reliably comparable across
candidates because each candidate pair can use different selected frames and a
different VGGT reconstruction.

Shared-geometry implementation:

- localize reference object and all candidates together per frame,
- select candidate/reference co-visible frames,
- run one shared VGGT reconstruction for the question,
- extract point clouds for all objects in the same coordinate system,
- compare reference-to-candidate distances in that shared point cloud.

Small tuning result on the first 5 samples:

| Strategy | Samples | Correct |
| --- | ---: | ---: |
| shared geometry, p90, 2 frames/candidate, max 8 frames | 5 | 1 |
| shared geometry, p75, 2 frames/candidate, max 8 frames | 5 | 2 |
| shared geometry, p75, 1 frame/candidate, max 4 frames | 5 | 2 |

Current default for relative distance:

- `frames_per_candidate=1`
- `max_compare_frames=4`
- `pointcloud_aggregate=p75`

Current limitation:

- Even with shared VGGT coordinates, the first 5-sample result is only 2/5.
- Remaining failures appear to come from localization/mask quality and the
  distance statistic, not option parsing.

Next optimization:

- add filters for duplicated/overlapping boxes,
- reject candidates with poor mask/point-cloud support,
- try center-to-center and robust closest-surface hybrids,
- run a 20-sample shared-geometry evaluation after those filters.
