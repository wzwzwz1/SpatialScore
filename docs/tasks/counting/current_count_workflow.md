# Current Object Counting Workflow

This note records the current VSI-Bench object-counting path in this repo.

## Runtime Path

1. VSI-Bench video is uniformly sampled to 64 frames.
2. For `object_counting`, the agent is prompted to call `CountVideoObjects3D` with only `objects`.
3. Runtime injects the sampled frame paths into the video counting tool.
4. `CountVideoObjects3D` runs Rex-Omni on each sampled frame to get 2D object boxes.
5. VGGT predicts per-frame 3D world points.
6. Each 2D detection is lifted to 3D by sampling points inside the bbox and taking a robust median 3D point.
7. Views of the same target object are clustered with constrained greedy clustering.
8. The final answer is the number of clustered object instances.

## Current Fixed Settings

- `video_counting_3d.num_frames`: `64`
- `video_counting_3d.cg_distance_threshold`: `0.7`
- `video_counting_3d.use_sam_masks`: `false`
- `video_counting_3d.use_tracking`: `false`
- `video_counting_3d.max_detections_per_frame`: `20`

## Current Validation

20 VSI-Bench `object_counting` samples, doc ids `0..19`, fixed threshold `0.7`:

- No SAM baseline: `12/20`, MAE `0.80`
- SAM2 masks only: `8/20`, MAE `1.05`
- SAM2 masks + tracking, first 5 samples only: `1/5`, MAE `1.20`

Conclusion: SAM2 tracking device mismatch is fixed, but SAM2 is not enabled by default because the current implementation reduced accuracy and is much slower.

## Paper Note: Absolute Distance

In the paper, absolute object distance is handled by `Get3DDistance`, not by subtracting object depths.

Paper workflow:

1. For object-distance questions, first localize the two objects.
2. Choose the closest point pair between the two objects, not simply bbox centers.
3. Call `Get3DDistance(image, point_1, point_2)`.
4. `Get3DDistance` uses MapAnything to reconstruct the 3D scene and returns the real-world 3D distance in meters.

The paper also states that `EstimateObjectDepth` returns distance from camera to object, which is a different quantity from distance between two objects.

Current repo status:

- `Get3DDistance` is now implemented with the same primitive interface: `image`, `point_1`, `point_2`.
- The current backend is VGGT world-points reconstruction, not MapAnything.
- LLM/tool orchestration still needs an upper-level object-to-object workflow to choose the two closest object points before calling `Get3DDistance`.
