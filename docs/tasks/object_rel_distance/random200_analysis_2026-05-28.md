# Object Rel Distance Random-200 Analysis

Date: 2026-05-28

## Run

- Task: `object_rel_distance`
- Workflow: 64-frame video sampling -> GroundingDINO localization -> SAM2 mask extraction -> VGGT 3D reconstruction -> closest-point distance ranking.
- Run directory: `/disk/wangzhe/SpatialScore/runs/object_rel_distance_random200_gpu2`
- Sample set: 200 fixed doc ids in `doc_ids.txt`.
- Dataset video state: after unzipping ScanNet and ScanNet++ videos, this run no longer fails from missing video files.

## Baseline Result

This is the baseline after switching the reference object to a single valid instance, but before the bbox-contact shortcut.

- Total: 200
- Evaluated successfully by tool: 171
- Correct: 73
- Wrong among evaluated: 98
- Tool errors: 29
- Accuracy over all sampled docs: 73 / 200 = 36.5%
- Accuracy over evaluated docs: 73 / 171 = 42.7%

The saved records use `ground_truth` as the answer field. Some quick inspection scripts that read `gt` will print `None`; do not interpret that as missing labels.

## Dataset Breakdown

| Dataset | Total | Evaluated | Correct | Wrong | Tool errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| arkitscenes | 46 | 40 | 17 | 23 | 6 |
| scannet | 76 | 66 | 28 | 38 | 10 |
| scannetpp | 78 | 65 | 28 | 37 | 13 |

The failure pattern is not concentrated in a single source dataset. All three datasets have similar evaluated accuracy.

## Tool Error Breakdown

| Error | Count |
| --- | ---: |
| No frames localized the reference object with any candidate object. | 17 |
| No finite point cloud for reference object. | 10 |
| No candidate object produced a valid shared 3D distance. | 2 |

These errors are mostly detection/mask/reconstruction availability failures rather than answer-ranking failures.

## Representative Cases

Exported failure materials are under `/disk/wangzhe/rel-distance/fail`.

- `doc_528`: wrong baseline answer caused by a multi-frame reference-object issue and a nearby false TV detection. User inspection showed frame 10 contains the real TV and table, while frame 21 contains a fake low-confidence TV. The baseline selected refrigerator; GT is table.
- `doc_3175`: reference point cloud missing for a small/weak reference object.
- `doc_3212`: no frame localized both reference and candidate objects.
- `doc_609`: no valid shared 3D candidate distance.
- `doc_794`: wrong ScanNet++ ranking example.

The reusable export script is `/disk/wangzhe/rel-distance/tools/export_rel_distance_case.py`.

## Finding From `doc_528`

For relative distance questions, the reference object is usually a single physical instance. Keeping all reference detections across frames can introduce false reference observations into frame selection or reconstruction. The current workflow therefore selects one valid reference instance by highest `(score, mask_pixels)` after SAM/VGGT validity checks.

`doc_528` exposed another special case: if the selected reference object and a candidate object are in strong 2D contact in the same frame, the answer can often be decided without trusting a wide-baseline VGGT reconstruction. In frame 10, the TV is on or overlapping the table, so the closest object is table.

Implemented patch:

- After selecting the reference instance, check candidate bboxes in the same frame.
- If a candidate strongly overlaps/touches the reference bbox, return that candidate directly with distance `0.0`.
- Store the decision in `payload.shortcut`.

Verification on `doc_528` after the patch:

- Prediction changed from `A refrigerator` to `C table`.
- Correct: `true`
- Selected reference frame: 10
- Reference score: 0.732
- Candidate: table
- Candidate score: 0.400
- IoU: 0.167
- Overlap against smaller bbox area: 0.349
- Edge gap ratio: 0.0

## Current Follow-Up Run

A 200-sample rerun with the bbox-contact shortcut is in progress:

- Run directory: `/disk/wangzhe/SpatialScore/runs/object_rel_distance_random200_gpu2_bbox_contact`
- Screen: `reldist200_bbox_gpu2`
- Process command is saved in `run_command.sh`.
- At the time this note was written, 9 / 200 docs had completed, with 5 correct and 6 shortcut activations. This is too early for a stable conclusion.

This run was stopped after inspection because it was using the first, overly broad shortcut rule. Do not treat it as a final evaluation.

Observed issue:

- At one checkpoint, shortcut-triggered samples were only about `10 / 23` correct.
- Later exported shortcut-false-positive cases included `doc_542`, `doc_543`, `doc_564`, `doc_567`, `doc_578`, `doc_588`, `doc_609`, `doc_612`, `doc_622`, `doc_794`, `doc_838`, `doc_850`, `doc_865`, `doc_1362`, `doc_1364`, `doc_1367`, and `doc_1379`.
- These cases were exported under `/disk/wangzhe/rel-distance/fail`.

Key diagnosis from manual inspection:

- 2D bbox contact alone is not enough.
- `doc_564` shows the failure clearly: sofa overlaps the fireplace bbox in 2D, but depth makes the TV closer in 3D.
- Low-confidence or tiny-overlap detections also triggered the old shortcut, for example `doc_865`.

Patch after this finding:

- Bbox contact is now only supporting evidence.
- The tool no longer forces the candidate distance to `0.0`.
- The normal 3D distance ranking is computed first.
- Contact is only applied when:
  - reference and candidate detection scores are both at least `0.40`,
  - bbox overlap is strong enough, not just edge adjacency,
  - and 3D near-distance statistics also support the contact candidate.
- The payload now records whether the contact evidence was applied or rejected via `shortcut.applied` and `shortcut.rejected_reason`.

Small validation run after tightening:

- Run directory: `/disk/wangzhe/SpatialScore/runs/object_rel_distance_shortcut_tight_check_gpu2`
- Docs: `528, 564, 865, 542, 543`
- Result: `3 / 5` correct.
- `doc_528`: remains correct, selected `table`; shortcut applied with 3D support.
- `doc_564`: fixed from wrong `sofa` to correct `tv`; shortcut no longer fires.
- `doc_542`: fixed from wrong `table` to correct `fireplace`; shortcut no longer fires.
- `doc_865`: improved from wrong `printer` to wrong `table`; shortcut no longer fires, but 3D still ranks `table` slightly closer than GT `ceiling light`.
- `doc_543`: remains wrong; shortcut no longer fires, but valid candidate coverage is still weak.

Next full evaluation should start a fresh run directory, because the old bbox-contact 200-run contains mixed old-rule results.

## Next Optimization Directions

- Analyze bbox-contact shortcut impact on the full 200-sample rerun.
- Inspect shortcut regressions, especially cases where overlapping boxes are not physical contact.
- Improve frame grouping for VGGT: very wide, non-contiguous selected frames can produce poor reconstruction.
- Consider candidate-specific local reconstruction groups when reference and candidate co-occur in a reliable frame.
- Keep changes sample-driven: add or reject rules based on exported real cases, not abstract intuition.
