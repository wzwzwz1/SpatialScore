# ViSRA-Style Video Counting Model Weights

This note records the weights needed to run the `CountVideoObjects3D` workflow:

`Rex-Omni -> SAM2 -> VGGT -> constrained greedy clustering`

The commands below assume the repository lives at `/disk/wangzhe/SpatialScore` and the conda
environment is named `SpatialScore`. Adjust paths on a new server as needed.

## 1. Environment

```bash
cd /disk/wangzhe/SpatialScore
conda activate SpatialScore

export HF_HOME=/home/wangzhe/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
```

If the mirror is unstable, unset `HF_ENDPOINT` and download from Hugging Face directly:

```bash
unset HF_ENDPOINT
```

## 2. Rex-Omni

Clone the Rex-Omni code, because SpatialScore imports its local `rex_omni` package:

```bash
git clone https://github.com/IDEA-Research/Rex-Omni.git /disk/wangzhe/Rex-Omni
```

Download the model snapshot into the Hugging Face cache:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="IDEA-Research/Rex-Omni",
    repo_type="model",
    resume_download=True,
)
print(path)
PY
```

Verify local availability:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

for name in [
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]:
    print(hf_hub_download("IDEA-Research/Rex-Omni", name, local_files_only=True))
PY
```

Expected disk usage is about 7.6 GB.

## 3. VGGT-1B

SpatialScore is configured to load VGGT from a local checkpoint path. Create the directory:

```bash
mkdir -p /disk/wangzhe/models/VGGT-1B
```

Download `model.pt` directly:

```bash
curl -L -C - --fail --progress-bar \
  https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt \
  -o /disk/wangzhe/models/VGGT-1B/model.pt

curl -L -C - --fail --progress-bar \
  https://huggingface.co/facebook/VGGT-1B/resolve/main/config.json \
  -o /disk/wangzhe/models/VGGT-1B/config.json
```

The `-C -` flag enables resume. Expected disk usage is about 4.7 GB.

Verify loading:

```bash
python - <<'PY'
from spatial_agent.tools.backends import get_vggt_backend

backend = get_vggt_backend(
    model_id="facebook/VGGT-1B",
    checkpoint_path="/disk/wangzhe/models/VGGT-1B/model.pt",
    device="cuda",
)
print(type(backend["model"]).__name__)
PY
```

## 4. SAM2

For `CountVideoObjects3D`, SAM2 can be loaded from Hugging Face:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="facebook/sam2.1-hiera-large",
    repo_type="model",
    resume_download=True,
)
print(path)
PY
```

For the older CountVid fallback pipeline, the repo currently expects CountVid-style local checkpoints:

```text
/disk/wangzhe/CountVid/checkpoints/sam2.1_hiera_large.pt
/disk/wangzhe/CountVid/checkpoints/countgd_box.pth
```

Those are not required for Rex-Omni + SAM2 + VGGT + CG clustering, but they are required if you want
`CountVideoObjects` fallback to run the CountVid subprocess path.

## 5. Tool Config

Update `configs/tool_config.server.json` on the new server:

```json
"video_counting_3d": {
  "num_frames": 64,
  "device": "cuda",
  "preprocess_mode": "pad",
  "rex_model_path": "IDEA-Research/Rex-Omni",
  "rex_backend": "transformers",
  "rex_repo_path": "/disk/wangzhe/Rex-Omni",
  "vggt_model_id": "facebook/VGGT-1B",
  "vggt_checkpoint_path": "/disk/wangzhe/models/VGGT-1B/model.pt",
  "sam2_model_id": "facebook/sam2.1-hiera-large",
  "use_sam_masks": true,
  "use_tracking": true,
  "tracking_max_frames": 50,
  "tracking_absent_patience": 2,
  "cg_distance_threshold": 0.35,
  "bbox_point_stride": 8,
  "max_detections_per_frame": 20
}
```

## 6. Smoke Test

Use a tiny two-frame smoke before running full 64-frame VSI-Bench:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python - <<'PY'
from pathlib import Path
from spatial_agent.runtime.config import SpatialAgentConfig, load_tool_config
from spatial_agent.tools.video_counting_3d import CountVideoObjects3DTool

frames = sorted(str(p) for p in Path("/tmp/spatial_agent_runs/sampled_frames/vsibench/test/0").glob("*.jpg"))[:2]
config = SpatialAgentConfig(
    artifact_dir="/tmp/spatial_agent_visra_real_smoke",
    tool_config=load_tool_config("/disk/wangzhe/SpatialScore/configs/tool_config.server.json"),
)
config.tool_config["video_counting_3d"] = dict(config.tool_config["video_counting_3d"])
config.tool_config["video_counting_3d"].update({
    "num_frames": 2,
    "use_sam_masks": False,
    "use_tracking": False,
    "max_detections_per_frame": 5,
    "bbox_point_stride": 16,
    "max_tokens": 1024,
})

result = CountVideoObjects3DTool(config).invoke(images=frames, objects=["table"])
print(result["status"], result.get("error"))
print(result.get("payload", {}).get("pipeline_stats"))
print(result.get("artifacts"))
PY
```

On a 20 GB RTX 3080, keep the GPU mostly free. The pipeline loads Rex-Omni and VGGT in stages, but
full 64-frame runs with SAM2 tracking still need substantial free memory.
