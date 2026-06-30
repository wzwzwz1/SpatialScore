# CountVid Video Counting Workflow Design

## 1. Goal and Scope

### Goal

Replace the current room-level video counting behavior:

- `LLM -> CountObjects(frame_1/frame_2/frame_3) -> LLM summarize`

with a dedicated **CountVid-style video counting workflow** that counts **unique object instances across a video**, rather than relying on the LLM to infer de-duplication from a few per-frame counts.

### Why this change

The current workflow is now stable enough at the orchestration layer, but it still fails at the counting layer:

- single-frame `CountObjects` often returns one visible instance per frame
- multiple frame-level counts are not enough to infer room-level count
- the agent currently has no explicit cross-frame instance de-duplication
- the LLM is forced to guess the room-level count from weak evidence

CountVid is a better fit for this task because it treats video counting as:

1. open-world per-frame counting / detection
2. temporal filtering
3. video-level instance propagation and unique-track counting

### Scope

This design covers:

- backend/tooling changes in `spatial_agent`
- LangGraph routing changes for counting tasks
- artifact and trace changes for debugging
- evaluation integration touchpoints

This design does **not** cover:

- frontend UI changes
- Spring Boot API integration
- training or fine-tuning CountVid modules
- exemplar-conditioned counting in the first version

## 2. Reference Material

### Primary references

- SpatialScore paper:
  - `/disk/wangzhe/SpatialScore/docs/references/papers/SpatialScore - Wu et al. 2026.pdf`
- CountVid paper:
  - [Open-World Object Counting in Videos (arXiv)](https://arxiv.org/abs/2506.15368)
- CountVid project:
  - [CountVid GitHub](https://github.com/niki-amini-naieni/CountVid)
  - [Oxford project page](https://www.robots.ox.ac.uk/~vgg/research/countvid/)

### Key CountVid insight to adopt

Do **not** let the LLM perform cross-frame de-duplication implicitly.

Instead:

1. generate per-frame candidate instances
2. filter temporally
3. propagate instances through the video
4. count **unique tracks / unique propagated instances**

## 3. Current Problem Summary

### Current counting behavior

For `vsibench_question_type == object_counting` and `input_modality == video`, the system currently:

1. samples `video_num_frames` frames from the video
2. selects three representative frames
3. calls single-image `CountObjects` on those representative frames
4. asks the LLM to produce the final answer

### Failure mode

This workflow is vulnerable when:

- each representative frame only exposes one visible instance
- instances appear from different viewpoints across frames
- different tables become visible in different frames
- the LLM sees repeated `instance_count=1` observations and concludes the room contains one object

### Root cause

The current system has:

- no video-level object identity
- no explicit track construction
- no room-level de-duplication

The orchestration is no longer the main blocker. The missing piece is a **video-native counting tool/workflow**.

## 4. Recommended Architecture

## 4.1 Recommended approach

Introduce a new video-native counting tool:

- `CountVideoObjects`

Keep existing single-image `CountObjects` unchanged.

### Why a new tool is preferred

This is better than composing several small tools with LLM control because CountVid is a tightly coupled pipeline:

- frame-level candidate generation
- temporal filtering
- propagation/tracking
- unique-instance aggregation

These stages should be executed deterministically by code, not improvised by the LLM step-by-step.

## 4.2 Routing strategy

### Default routing

- if `input_modality != "video"` and counting task:
  - prefer `CountObjects`
- if `input_modality == "video"` and counting task:
  - prefer `CountVideoObjects`

### Compatibility fallback

Version 1 should support explicit fallback behavior:

- if `CountVideoObjects` is unavailable:
  - return `unavailable`
  - do **not** silently fall back to multi-frame `CountObjects`

This keeps failure modes visible during rollout.

## 5. Workflow Design

## 5.1 Input / Output contract

### Tool name

- `CountVideoObjects`

### Input schema

```json
{
  "images": ["image-0", "image-1", "..."],
  "objects": "table"
}
```

Accepted runtime forms:

- `objects: string`
- `objects: string[]`
- `images: string[]`

For version 1, the tool should operate on sampled frames already produced by the existing pipeline.

### Output schema

```json
{
  "instance_count": 4,
  "tracks": [
    {
      "track_id": "table_000",
      "object": "table",
      "supporting_frames": ["image-0", "image-8", "image-15"],
      "supporting_points": [[0.10, 0.30], [0.18, 0.28], [0.22, 0.31]]
    }
  ],
  "frame_summaries": [
    {
      "image": "image-0",
      "candidate_count": 2,
      "filtered_count": 2
    }
  ],
  "backend": "countvid:countgd_box+sam2.1",
  "artifact_descriptions": []
}
```

### Required semantics

- `instance_count` must be the **video-level unique instance count**
- `tracks` must reflect unique propagated instances, not raw per-frame detections
- `frame_summaries` are for debugging and analysis

## 5.2 Internal pipeline stages

### Stage 1: Candidate generation

Goal:

- detect or count target objects on each frame
- produce prompts usable for downstream propagation

Backend strategy:

- adopt CountVid’s frame-level counting/detection module
- first target is **CountGD-Box**

Outputs per frame:

- candidate boxes and/or points
- confidence values
- normalized coordinates

### Stage 2: Temporal filtering

Goal:

- remove one-frame spikes and unstable false positives

Version 1 behavior:

- apply simple window-based support filtering across neighboring sampled frames
- discard isolated detections unsupported in adjacent frames

Longer-term behavior:

- match CountVid’s more faithful temporal filtering pipeline

### Stage 3: Video propagation / tracking

Goal:

- propagate candidate instances through the sampled frame sequence
- assign persistent identities to unique objects

Backend strategy:

- adopt CountVid’s SAM 2 / SAM 2.1 promptable video propagation approach

Outputs:

- video-level object tracks
- per-track supporting frames
- per-track spatial evidence

### Stage 4: Unique instance aggregation

Goal:

- count unique objects after propagation, not raw detections

Aggregation rule:

- `instance_count = number of unique accepted tracks`

Version 1 acceptance rule:

- a valid track must be supported by at least one propagated object instance
- configurable minimum support threshold may be added later

## 6. LangGraph Integration

## 6.1 State additions

No mandatory global state changes are required for the first version if `CountVideoObjects` is implemented as an atomic tool.

Optional future state fields:

- `video_counting_summary`
- `video_counting_tracks`
- `video_counting_debug`

## 6.2 Node behavior

### `reason_node`

No structural change required.

The LLM should be able to select:

- `CountVideoObjects` for video counting
- `CountObjects` for image counting

### `route_node`

Add counting-specific preference rule to the prompt and available-tool set, but avoid hard-coding tool substitution in the node itself.

### `observe_node`

No special queue logic is required if `CountVideoObjects` is atomic.

This is an important simplification:

- instead of `CountObjects -> CountObjects -> CountObjects -> finish`
- we want `CountVideoObjects -> finish`

### `finalize_node`

Preserve current counting answer normalization:

- final answer must still reduce to a pure Arabic numeral for VSI-Bench counting tasks

## 7. File Structure Proposal

### New files

- `spatial_agent/tools/video_counting.py`
  - `CountVideoObjectsTool`
- `spatial_agent/tools/countvid_backend.py`
  - CountVid-specific loading and inference wrapper
- `spatial_agent/tools/video_counting_utils.py`
  - filtering, track formatting, artifact helpers

### Updated files

- `spatial_agent/tools/registry.py`
  - register `CountVideoObjects`
- `spatial_agent/adapters/prompting.py`
  - tell the LLM to prefer `CountVideoObjects` for video counting
- `spatial_agent/prompts/react_system_prompt.py`
  - tool description update
- `spatial_agent/graph/tool_args.py`
  - support automatic `images` binding for `CountVideoObjects`
- `spatial_agent/analysis/report.py`
  - expose track-level and frame-summary outputs
- `tests/spatial_agent/test_tools.py`
- `tests/spatial_agent/test_react_graph.py`

## 8. Tool Configuration Design

Add a new config section:

```json
{
  "video_counting": {
    "backend": "countvid",
    "countgd_repo_path": "/path/to/CountVid",
    "countgd_checkpoint_path": "/path/to/countgd_box.pth",
    "sam2_checkpoint_path": "/path/to/sam2.1_hiera_large.pt",
    "sam2_config_name": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "device": "cuda",
    "window_size": 3,
    "min_track_support": 1
  }
}
```

Aliases to accept:

- `video_counting`
- `countvid`

## 9. Prompting Strategy

## 9.1 System prompt rule

Add explicit guidance:

- for video counting tasks, prefer `CountVideoObjects`
- for image counting tasks, prefer `CountObjects`

Example wording:

> For video counting questions, prefer `CountVideoObjects` because it counts unique object instances across frames. Use `CountObjects` for single-image counting.

## 9.2 Why this matters

This keeps the LLM from recreating a brittle multi-frame counting strategy using repeated single-image calls when a video-native counting tool is available.

## 10. Failure and Fallback Strategy

### Failure types

1. CountVid backend unavailable
2. CountGD-Box inference failure
3. SAM propagation failure
4. empty candidate set

### Version 1 behavior

- initialization failure:
  - return `unavailable`
- runtime failure:
  - return `error`
- empty but valid result:
  - return `success` with `instance_count = 0`

### Important rule

Do **not** silently fall back to repeated `CountObjects` in version 1.

The goal of this rollout is to make the CountVid path observable and debuggable.

## 11. Artifacts and Debuggability

The tool should emit artifacts that help debug track quality.

Recommended artifact types:

1. frame-level candidate visualization
2. temporal filtering summary
3. propagated track overlay
4. track manifest JSON

Example artifact payload entry:

```json
{
  "path": "/.../countvid_tracks.json",
  "kind": "track_manifest",
  "description": "Unique propagated object tracks used for final video-level counting."
}
```

## 12. Testing Plan

## 12.1 Unit tests

### `CountVideoObjectsTool`

- returns structured success payload
- returns `unavailable` when backend init fails
- returns `error` on inference failure
- handles zero-instance success case

### Aggregation / track formatting

- stable track formatting
- correct unique count from synthetic track inputs
- temporal filter removes unsupported single-frame spikes

## 12.2 Graph integration tests

- for video counting tasks, prompt/tool choice includes `CountVideoObjects`
- `CountVideoObjects` receives sampled frames in time order
- successful call can directly finalize with numeric answer
- graph does not require repeated `CountObjects` when `CountVideoObjects` exists

## 12.3 Regression tests

- image counting still uses `CountObjects`
- existing multi-step ReAct logic remains valid for non-video-counting workflows

## 13. Rollout Plan

### Phase 1

- add config scaffolding
- add placeholder backend wrapper
- add `CountVideoObjectsTool` with mockable interface
- integrate into registry and prompting

### Phase 2

- wire actual CountVid modules
- emit debug artifacts
- validate on a small VSI-Bench counting slice

### Phase 3

- compare:
  - current `CountObjects`-based video counting
  - CountVid workflow
- inspect gains on room-level counting samples

## 14. Recommendation

Recommended implementation path:

1. keep `CountObjects` as the single-image counting tool
2. add `CountVideoObjects` as a dedicated video counting tool
3. route video counting tasks to `CountVideoObjects`
4. let the LLM focus on tool selection and final phrasing, not cross-frame de-duplication

This is the cleanest way to incorporate CountVid while preserving the existing SpatialAgent architecture.
