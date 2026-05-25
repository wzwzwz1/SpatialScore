from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from spatial_agent.tools.backends import (
    artifact_dir_for_tool,
    ensure_image_paths,
    ensure_object_names,
    get_tool_settings,
)
from spatial_agent.tools.base import BaseSpatialTool
from spatial_agent.tools.countvid_backend import (
    generate_candidates_countgd,
    get_countvid_backend,
    run_sam2_propagation,
)
from spatial_agent.tools.video_counting_utils import (
    aggregate_unique_tracks,
    build_frame_summaries,
    build_track_payload,
    save_candidate_overlay,
    save_track_overlay,
    temporal_window_filter,
)


class CountVideoObjectsTool(BaseSpatialTool):
    name = "CountVideoObjects"
    description = (
        "Count unique object instances across a video using cross-frame propagation. "
        "Use this tool for video counting tasks instead of repeated single-frame CountObjects calls. "
        "It detects candidates per frame, filters temporally, propagates instances through the video, "
        "and returns the video-level unique instance count."
    )
    args_schema = {
        "type": "object",
        "properties": {
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered sampled video frame paths (time order).",
            },
            "objects": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "description": "Target object category name(s) to count.",
            },
        },
        "required": ["objects"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config

    def invoke(self, **kwargs) -> Dict[str, Any]:
        image_paths = ensure_image_paths(kwargs.get("images") or kwargs.get("image"))
        objects = ensure_object_names(kwargs.get("objects"))
        if not image_paths:
            return self.error("CountVideoObjects requires at least one image/frame path.")
        if not objects:
            return self.error("CountVideoObjects requires one or more object names.")

        settings = get_tool_settings(self.config, self.name, aliases=["video_counting", "countvid"])
        countgd_repo = settings.get("countgd_repo_path")
        countgd_ckpt = settings.get("countgd_checkpoint_path")
        sam2_ckpt = settings.get("sam2_checkpoint_path")
        sam2_config = settings.get("sam2_config_name")
        window_size = int(settings.get("window_size", 3))
        min_track_support = int(settings.get("min_track_support", 1))
        device = settings.get("device", "cuda")

        backend = None
        backend_label = "countvid:unavailable"
        try:
            backend = get_countvid_backend(
                countgd_repo_path=str(countgd_repo) if countgd_repo else "",
                countgd_checkpoint_path=str(countgd_ckpt) if countgd_ckpt else "",
                sam2_checkpoint_path=str(sam2_ckpt) if sam2_ckpt else "",
                sam2_config_name=str(sam2_config) if sam2_config else "",
                device=str(device),
            )
        except Exception as exc:
            return self.unavailable(f"CountVid backend initialization failed: {exc}")

        countgd_ok = backend.get("countgd_available", False)
        sam2_ok = backend.get("sam2_available", False)

        if not countgd_ok or not sam2_ok:
            missing = []
            if not countgd_ok:
                missing.append("CountGD-Box")
            if not sam2_ok:
                missing.append("SAM2.1")
            return self.unavailable(
                f"CountVid backend partially unavailable: {', '.join(missing)} not available. "
                f"Check countgd_repo_path, countgd_checkpoint_path, sam2_checkpoint_path, sam2_config_name in tool_config."
            )

        backend_label = "countvid:countgd_box+sam2.1"

        # Stage 1: Candidate generation
        try:
            frame_detections = generate_candidates_countgd(backend, image_paths, objects)
        except Exception as exc:
            return self.error(f"CountGD-Box candidate generation failed: {exc}")

        # Stage 2: Temporal filtering
        try:
            frame_detections = temporal_window_filter(frame_detections, window_size=window_size)
        except Exception as exc:
            return self.error(f"Temporal filtering failed: {exc}")

        # Stage 3: Video propagation / tracking
        try:
            raw_tracks = run_sam2_propagation(backend, image_paths, frame_detections, objects)
        except Exception as exc:
            return self.error(f"SAM 2.1 propagation failed: {exc}")

        # Stage 4: Unique instance aggregation
        image_aliases = [f"image-{i}" for i in range(len(image_paths))]
        accepted_tracks = aggregate_unique_tracks(raw_tracks, min_track_support=min_track_support)
        formatted_tracks = build_track_payload(accepted_tracks, image_aliases)
        frame_summaries = build_frame_summaries(frame_detections, image_aliases)
        instance_count = len(accepted_tracks)

        # Artifacts
        artifact_dir = artifact_dir_for_tool(self.config, self.name)
        artifacts: List[str] = []

        track_overlay_path = artifact_dir / "track_overlay.png"
        try:
            artifacts.append(
                save_track_overlay(image_paths, accepted_tracks, track_overlay_path)
            )
        except Exception:
            pass

        candidate_overlay_path = artifact_dir / "candidate_overlay.png"
        try:
            artifacts.append(
                save_candidate_overlay(image_paths, frame_detections, candidate_overlay_path)
            )
        except Exception:
            pass

        # Track manifest JSON artifact
        import json
        manifest_path = artifact_dir / "countvid_tracks.json"
        try:
            manifest_path.write_text(
                json.dumps(
                    {
                        "instance_count": instance_count,
                        "tracks": formatted_tracks,
                        "frame_summaries": frame_summaries,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            artifacts.append(str(manifest_path))
        except Exception:
            pass

        payload = {
            "instance_count": instance_count,
            "tracks": formatted_tracks,
            "frame_summaries": frame_summaries,
            "backend": backend_label,
            "artifact_descriptions": [
                {
                    "path": track_overlay_path,
                    "kind": "track_overlay",
                    "description": "Propagated object tracks overlaid on sampled video frames.",
                },
                {
                    "path": candidate_overlay_path,
                    "kind": "candidate_overlay",
                    "description": "Per-frame candidate detections with temporal filtering results.",
                },
                {
                    "path": str(manifest_path),
                    "kind": "track_manifest",
                    "description": "Unique propagated object tracks used for final video-level counting.",
                },
            ],
        }
        return self.success(payload=payload, artifacts=artifacts)
