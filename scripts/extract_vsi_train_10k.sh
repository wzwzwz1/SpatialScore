#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/disk/wangzhe/VSI-Train-10k}"
JOBS="${JOBS:-3}"

cd "${ROOT}"

if ! command -v zstd >/dev/null 2>&1; then
  echo "zstd is required to extract *.tar.zst shards." >&2
  exit 1
fi

missing=0
for i in $(seq -f "%03g" 0 8); do
  shard="vsi_train_shard_${i}.tar.zst"
  if [ ! -s "${shard}" ]; then
    echo "missing shard: ${ROOT}/${shard}" >&2
    missing=1
  fi
done
if [ "${missing}" -ne 0 ]; then
  exit 1
fi

echo "Extracting VSI-Train-10k shards under ${ROOT} with ${JOBS} parallel jobs"

printf '%s\n' vsi_train_shard_*.tar.zst | xargs -P "${JOBS}" -I {} bash -lc '
  set -euo pipefail
  shard="$1"
  echo "[$(date "+%F %T")] extracting ${shard}"
  zstd -d "${shard}" -c | tar -xf -
  echo "[$(date "+%F %T")] finished ${shard}"
' _ {}

echo "Extraction finished"
