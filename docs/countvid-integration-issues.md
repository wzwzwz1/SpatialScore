# CountVid 集成问题记录

## 1. 依赖与编译问题

### 1.1 GroundingDINO CUDA ops 编译失败
- **现象**：`ModuleNotFoundError: No module named 'MultiScaleDeformableAttention'`
- **原因**：CountVid 依赖自定义 CUDA kernel，需要从源码编译。服务器上缺少 `spaces`、`yapf`、`addict`、`pycocotools` 等包
- **解决**：安装缺失包后编译，但遇到 PyTorch 2.6 API 不兼容

### 1.2 PyTorch 2.6 API 不兼容
- **现象**：CUDA 编译报错 `no suitable conversion function from "const at::DeprecatedTypeProperties" to "c10::ScalarType"`
- **原因**：PyTorch 2.6 中 `tensor.type()` 返回 `DeprecatedTypeProperties`，不能隐式转换
- **解决**：手动 patch CountVid 源码
  - `value.type()` → `value.scalar_type()`（AT_DISPATCH_FLOATING_TYPES 上下文）
  - `tensor.type().is_cuda()` → `tensor.is_cuda()`

### 1.3 torch.load 默认参数变化
- **现象**：`_pickle.UnpicklingError: Weights only load failed`
- **原因**：PyTorch 2.6 默认 `weights_only=True`，CountVid checkpoint 包含非权重数据
- **解决**：`torch.load(..., weights_only=False)`

### 1.4 huggingface-hub 版本冲突
- **现象**：`ImportError: huggingface-hub>=0.30.0,<1.0 is required but found 1.16.1`
- **原因**：transformers 4.51.3 要求 huggingface-hub<1.0，环境中装了 1.16.1
- **解决**：`pip install "huggingface-hub<1.0,>=0.30.0"` 降级到 0.36.2

### 1.5 LD_LIBRARY_PATH 缺失
- **现象**：`ImportError: libc10.so: cannot open shared object file`
- **原因**：编译的 MultiScaleDeformableAttention.so 依赖 torch 的动态库，但不在 LD_LIBRARY_PATH 中
- **解决**：subprocess 中注入 `LD_LIBRARY_PATH=<torch_lib_path>`

---

## 2. 架构与接入问题

### 2.1 无法 in-process 使用 CountVid
- **现象**：`count_in_videos.py` 作为脚本在 import 时执行大量副作用代码（`device = torch.device("cuda")`、`args = parser.parse_args()` 等），无法安全 import
- **解决**：改用 subprocess 方式，直接调用 `count_in_videos.py` 作为子进程

### 2.2 subprocess 中 Temporal Filter 不可用
- **现象**：启用 `--temporal_filter` 后结果异常（32 帧从 26 变为 1）
- **原因**：`temporal_filter_fast()` 使用 `multiprocessing.Pool()`，在 subprocess 环境中可能 fork 失败或状态异常
- **状态**：暂时关闭 temporal filter，需要进一步调试

### 2.3 T.json 输出路径不一致
- **现象**：tracks 解析为 0
- **原因**：CountVid 将 `T.json` 保存在当前工作目录（cwd=repo_path），而非 temp 目录
- **解决**：同时检查 `tmpdir/T.json` 和 `repo_path/T.json` 两个路径

---

## 3. 计数准确性问题

### 3.1 帧数与 over-counting 的 trade-off
- **现象**：

  | 帧数 | instance_count | Ground Truth | 问题 |
  |---|---|---|---|
  | 8 | 3 | 4 | 接近，差 1 |
  | 12 | 3 | 4 | 同 8 帧 |
  | 16 | 10 | 4 | 开始 over-counting |
  | 32 | 26 | 4 | 严重 over-counting |

- **根因**：CountVid 的 `get_new_objs` 机制——仅在每帧检查候选是否被现有 tracks 覆盖，未覆盖则创建新 track。帧数越多，CountGD 误检（把非 table 的物体检测为 table）累积越多，每个误检都变成独立 track
- **解决方向**：Phase 4 — Unique Instance Aggregation，对相似 tracks 做合并

### 3.2 跨帧 track 去重缺失（Phase 4 未完成）
- **现象**：同一 table 从不同帧被多次初始化为不同 track（如 track_4 到 track_10 都在 frame-14 被发现为"新对象"）
- **根因**：没有 track merge 逻辑——raw tracks 直接等于 unique tracks
- **需要的逻辑**：
  1. 计算 track 间相似度（temporal overlap + spatial proximity + mask IoU）
  2. 构建相似图，连通分量聚类
  3. 每个 cluster 合并为一个 canonical unique track

### 3.3 8 帧差 1 的原因
- **现象**：8 帧找到 3 个 table，Ground Truth 是 4
- **可能原因**：
  - 第 4 个 table 在采样帧中从未被 CountGD 检测到（遮挡、角度问题）
  - 或者检测到了但 SAM2 mask 与已有 track 重叠被过滤了
  - 需要查看具体帧的检测结果才能定位

---

## 4. 工程与测试问题

### 4.1 网络限制
- 服务器无法直连 huggingface.co 和 Google Drive
- BERT 下载需要 HTTP 代理且修复 `HF_ENDPOINT` 环境变量
- CountGD-Box checkpoint（955MB）从 Google Drive 下载耗时约 40 分钟

### 4.2 SOCKS 代理冲突
- `ALL_PROXY=socks5h://...` 导致 httpx/OpenAI client 初始化失败
- 需要每次运行时 `env -u ALL_PROXY -u all_proxy ...` 清除

### 4.3 recursion_limit 不足
- LangGraph 默认 `recursion_limit=25`，视频计数多步 ReAct 循环超出限制
- 解决：提升到 100

### 4.4 max_repairs 不足
- 视频计数要求至少观察 3 帧才能 finish，LLM 提前 finish 触发 repair
- 默认 `max_repairs=2` 不够
- 解决：提升到 3

---

## 5. 下一步优先级

1. **Phase 4（Unique Instance Aggregation）**— 解决 over-counting，预期能将 32 帧的 26 tracks 合并到接近 4
2. **Subprocess temporal filter 调试**— 修复 multiprocessing 问题后可过滤 transient false positives
3. **Phase 5 可观测性完善**— seed_overlay、统计字段，帮助定位为什么 8 帧差 1
