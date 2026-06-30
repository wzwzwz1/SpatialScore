#!/usr/bin/env bash
set -euo pipefail

cd /disk/wangzhe/SpatialScore

PYTHON_BIN="${PYTHON_BIN:-/home/wangzhe/miniconda3/envs/SpatialScore/bin/python}"
TRAIN_ROOT="${TRAIN_ROOT:-/disk/wangzhe/VSI-Train-10k}"
ANNOTATIONS="${ANNOTATIONS:-${TRAIN_ROOT}/vsi_train_10k.parquet}"
LIMIT="${LIMIT:-20}"
OUT="${OUT:-runs/vsi_train_abs_distance_limit${LIMIT}_gpu${CUDA_VISIBLE_DEVICES:-unknown}}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434/api/generate}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5vl:7b-fp16}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"

mkdir -p "${OUT}"

exec "${PYTHON_BIN}" scripts/eval_vsibench_object_abs_distance.py \
  --annotations-path "${ANNOTATIONS}" \
  --video-root "${TRAIN_ROOT}" \
  --question-type absolute_distance \
  --dataset-cache-dir "${TRAIN_ROOT}" \
  --tool-config-path configs/tool_config.server.json \
  --output-dir "${OUT}" \
  --limit "${LIMIT}" \
  --num-frames 64 \
  --method mask_pointcloud_multiframe \
  --pointcloud-aggregate p90 \
  --top-distance-frames 0 \
  --single-object-frames 2 \
  --bridge-frames 5 \
  --max-vggt-frames 0 \
  --enable-instance-verifier \
  --enable-instance-verifier-recheck \
  --instance-verifier-recheck-all-rounds \
  --enable-instance-verifier-best-match-fallback \
  --enable-final-frame-verifier \
  --verifier-backend ollama_generate \
  --verifier-model "${OLLAMA_MODEL}" \
  --verifier-ollama-url "${OLLAMA_URL}" \
  --resume \
  "$@"
