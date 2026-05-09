# SpatialAgent 服务器侧 `tool_config` 模板

这份文档给你一份可以直接在服务器上改路径的 `tool_config` 模板，覆盖当前已经接入的主要空间工具：

- RAFT
- Depth Anything V2
- SAM2
- VGGT
- Orient Anything
- Grounding DINO

## 1. 推荐放法

建议你在服务器上准备一个 Python 配置文件，或者直接复制仓库里的 JSON 模板：

- [`/disk/wangzhe/SpatialScore/docs/tool_config.server.template.json`](/disk/wangzhe/SpatialScore/docs/tool_config.server.template.json)

如果你希望在 Python 里构造，也可以这样写：

```python
from spatial_agent.runtime.config import SpatialAgentConfig

config = SpatialAgentConfig(
    llm_backend="hf",
    qwen_model_path="/data/models/Qwen2.5-VL-7B-Instruct",
    artifact_dir="/data/spatial_agent_runs",
    tool_config={
        "raft": {
            "checkpoint_path": "/data/models/raft/raft-things.pth",
            "small": False,
            "mixed_precision": True,
            "alternate_corr": False,
            "iters": 20,
            "device": "cuda",
        },
        "depth": {
            "checkpoint_path": "/data/models/depth_anything/depth_anything_v2_metric_hypersim_vitl.pth",
            "encoder": "vitl",
            "max_depth": 20,
            "input_size": 518,
            "device": "cuda",
        },
        "mask": {
            "model_id": "facebook/sam2.1-hiera-large",
            "device": "cuda",
        },
        "camera": {
            "hf_model_id": "facebook/VGGT-1B",
            "preprocess_mode": "pad",
            "device": "cuda",
        },
        "orientation": {
            "checkpoint_repo_id": "Viglong/Orient-Anything",
            "checkpoint_filename": "croplargeEX2/dino_weight.pt",
            "dino_mode": "large",
            "remove_background": True,
            "device": "cuda",
        },
        "localization": {
            "model_id": "IDEA-Research/grounding-dino-base",
            "box_threshold": 0.30,
            "text_threshold": 0.25,
            "enable_ram_tags": False,
            "device": "cuda",
        },
        "motion": {
            "hf_model_id": "facebook/VGGT-1B",
            "preprocess_mode": "pad",
            "device": "cuda",
        },
    },
)
```

## 2. 如果你想用本地 checkpoint 而不是 Hugging Face model id

### 2.1 SAM2

如果你不想从 Hugging Face 自动下载，而是服务器本地已经有权重和配置文件，可以这样写：

```python
"mask": {
    "checkpoint_path": "/data/models/sam2/sam2.1_hiera_large.pt",
    "config_path": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "device": "cuda",
}
```

注意这里的 `config_path` 走的是原仓库 `version_0/SpatialAgent/sam2/` 下的相对配置风格。

### 2.2 VGGT

如果你有本地 checkpoint：

```python
"camera": {
    "checkpoint_path": "/data/models/vggt/model.pt",
    "preprocess_mode": "pad",
    "device": "cuda",
},
"motion": {
    "checkpoint_path": "/data/models/vggt/model.pt",
    "preprocess_mode": "pad",
    "device": "cuda",
}
```

### 2.3 Orient Anything

如果你已经把权重下好了：

```python
"orientation": {
    "checkpoint_path": "/data/models/orient_anything/croplargeEX2/dino_weight.pt",
    "dino_mode": "large",
    "remove_background": True,
    "device": "cuda",
}
```

## 3. 可选增强：给 `LocalizeObjects` 打开 RAM tags

如果你想让定位工具顺带输出 RAM tags，可以再配一组 `ram`：

```python
tool_config = {
    "localization": {
        "model_id": "IDEA-Research/grounding-dino-base",
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "enable_ram_tags": True,
        "device": "cuda",
    },
    "ram": {
        "checkpoint_path": "/data/models/ram/ram_swin_large_14m.pth",
        "vit": "swin_l",
        "image_size": 384,
        "device": "cuda",
    },
}
```

注意：`RAM` 这条链除了 checkpoint，本地还需要原仓库里那套 tag embedding / threshold 文件能被正确访问。

## 4. 每个工具最关键的必填项

### `EstimateOpticalFlow`

至少要有：

```python
"raft": {
    "checkpoint_path": ".../raft-things.pth"
}
```

### `EstimateObjectDepth`

至少要有：

```python
"depth": {
    "checkpoint_path": ".../depth_anything_v2_metric_hypersim_vitl.pth",
    "encoder": "vitl"
}
```

### `GetObjectMask`

二选一：

```python
"mask": {
    "model_id": "facebook/sam2.1-hiera-large"
}
```

或者：

```python
"mask": {
    "checkpoint_path": "...",
    "config_path": "..."
}
```

### `GetCameraParametersVGGT`

二选一：

```python
"camera": {
    "hf_model_id": "facebook/VGGT-1B"
}
```

或者：

```python
"camera": {
    "checkpoint_path": "..."
}
```

### `GetObjectOrientation`

推荐至少有：

```python
"orientation": {
    "checkpoint_repo_id": "Viglong/Orient-Anything",
    "checkpoint_filename": "croplargeEX2/dino_weight.pt",
    "dino_mode": "large"
}
```

### `LocalizeObjects`

推荐至少有：

```python
"localization": {
    "model_id": "IDEA-Research/grounding-dino-base"
}
```

### `EstimateObjectMotion`

默认复用 VGGT，所以至少要有：

```python
"motion": {
    "hf_model_id": "facebook/VGGT-1B"
}
```

或者本地 checkpoint。

## 5. 和 VSI-Bench 一起跑时的建议

如果你是拿这个 agent 去跑 `VSI-Bench`，建议至少把下面这几条配好：

- `raft`
- `depth`
- `camera`
- `localization`
- `motion`

因为视频任务里，这些能力通常比单图 mask / orientation 更常被触发。

## 6. 你最常需要改的只有这些路径

如果只想最快跑通，通常你只需要改下面这些：

- `/data/models/Qwen2.5-VL-7B-Instruct`
- `/data/models/raft/raft-things.pth`
- `/data/models/depth_anything/depth_anything_v2_metric_hypersim_vitl.pth`
- `facebook/sam2.1-hiera-large`
- `facebook/VGGT-1B`
- `Viglong/Orient-Anything`
- `IDEA-Research/grounding-dino-base`

## 7. 相关文档

- SpatialAgent 主说明：
  - [`/disk/wangzhe/SpatialScore/spatial_agent/README.md`](/disk/wangzhe/SpatialScore/spatial_agent/README.md)
- VSI-Bench 评测说明：
  - [`/disk/wangzhe/SpatialScore/docs/spatial_agent_vsibench.md`](/disk/wangzhe/SpatialScore/docs/spatial_agent_vsibench.md)
- 可直接复制的 JSON 模板：
  - [`/disk/wangzhe/SpatialScore/docs/tool_config.server.template.json`](/disk/wangzhe/SpatialScore/docs/tool_config.server.template.json)
