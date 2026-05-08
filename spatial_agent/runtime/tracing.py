from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict


def write_trace(trace: Dict[str, Any], artifact_dir: str, task_id: str) -> str:
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    trace_path = path / f"{task_id}.json"
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
    return str(trace_path)

