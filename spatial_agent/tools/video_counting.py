from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from spatial_agent.tools.backends import (
    artifact_dir_for_tool,
    ensure_image_paths,
    ensure_object_names,
    get_tool_settings,
    load_pil_image,
)
from spatial_agent.tools.base import BaseSpatialTool
from spatial_agent.tools.countvid_backend import (
    generate_candidates_countgd,
    get_countvid_backend,
    run_countvid_subprocess,
    run_sam2_propagation,
)
from spatial_agent.tools.video_counting_utils import (
    aggregate_unique_tracks,
    build_frame_summaries,
    build_track_payload,
    save_track_overlay,
    save_unique_track_overlay,
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
        temporal_filter = bool(settings.get("temporal_filter", False))
        obj_batch_size = int(settings.get("obj_batch_size", 30))
        obj_batch_size_filter = int(settings.get("obj_batch_size_filter", 100))
        img_batch_size = int(settings.get("img_batch_size", 10))
        min_obj_area = int(settings.get("min_obj_area", 0))
        downsample_factor = float(settings.get("downsample_factor", 1.0))
        sample_frames = int(settings.get("sample_frames", 0))
        convert_to_rgb = bool(settings.get("convert_to_rgb", False))
        save_countgd_video = bool(settings.get("save_countgd_video", False))
        save_sam_indep_video = bool(settings.get("save_sam_indep_video", False))
        save_final_video = bool(settings.get("save_final_video", False))
        timeout_seconds = int(settings.get("timeout_seconds", 1200))
        apply_spatialscore_aggregation = bool(settings.get("apply_spatialscore_aggregation", False))

        # Use the subprocess-based approach (primary). It calls CountVid's
        # count_in_videos.py directly, avoiding complex in-process dependencies
        # (GroundingDINO custom CUDA ops, GCC, etc.).
        try:
            subprocess_result = run_countvid_subprocess(
                image_paths=image_paths,
                objects=objects,
                countgd_repo_path=str(countgd_repo) if countgd_repo else "",
                countgd_checkpoint_path=str(countgd_ckpt) if countgd_ckpt else "",
                sam2_checkpoint_path=str(sam2_ckpt) if sam2_ckpt else "",
                sam2_config_name=str(sam2_config) if sam2_config else "",
                device=str(device),
                window_size=window_size,
                temporal_filter=temporal_filter,
                obj_batch_size=obj_batch_size,
                obj_batch_size_filter=obj_batch_size_filter,
                img_batch_size=img_batch_size,
                min_obj_area=min_obj_area,
                downsample_factor=downsample_factor,
                sample_frames=sample_frames,
                convert_to_rgb=convert_to_rgb,
                save_countgd_video=save_countgd_video,
                save_sam_indep_video=save_sam_indep_video,
                save_final_video=save_final_video,
                timeout_seconds=timeout_seconds,
            )
        except FileNotFoundError:
            return self.unavailable(
                f"CountVid script not found at {countgd_repo}/count_in_videos.py"
            )
        except subprocess.TimeoutExpired:
            return self.error("CountVid subprocess timed out (20 min limit)")
        except Exception as exc:
            return self.error(f"CountVid subprocess failed: {exc}")

        raw_instance_count = subprocess_result["instance_count"]
        raw_tracks = subprocess_result["raw_tracks"]
        frame_summaries = subprocess_result["frame_summaries"]
        backend_label = subprocess_result["backend"]
        countvid_stats = {"official_pipeline": False, **subprocess_result.get("countvid_stats", {})}

        # Phase 4: Unique Instance Aggregation — merge raw tracks into canonical tracks
        point_threshold_px = float(settings.get("track_merge_point_threshold_px", 50))
        merge_overlap_min = int(settings.get("track_merge_overlap_min_frames", 2))

        image_aliases = [f"image-{i}" for i in range(len(image_paths))]
        image_sizes: List[Tuple[int, int]] = []
        for p in image_paths:
            try:
                img = load_pil_image(p)
                image_sizes.append(img.size)
            except Exception:
                image_sizes.append((1, 1))

        if apply_spatialscore_aggregation:
            accepted_tracks = aggregate_unique_tracks(
                raw_tracks,
                min_track_support=min_track_support,
                point_threshold_px=point_threshold_px,
                merge_overlap_min_frames=merge_overlap_min,
            )
        else:
            accepted_tracks = raw_tracks
        formatted_tracks = build_track_payload(accepted_tracks, image_aliases, image_sizes)
        instance_count = len(accepted_tracks) if apply_spatialscore_aggregation else raw_instance_count

        # Artifacts
        artifact_dir = artifact_dir_for_tool(self.config, self.name)
        artifacts: List[str] = []

        # Raw track overlay
        track_overlay_path = artifact_dir / "track_overlay.png"
        try:
            if raw_tracks:
                artifacts.append(
                    save_track_overlay(image_paths, raw_tracks, track_overlay_path)
                )
        except Exception:
            pass

        # Unique track overlay (color-coded per instance)
        unique_track_overlay_path = artifact_dir / "unique_track_overlay.png"
        try:
            if accepted_tracks:
                artifacts.append(
                    save_unique_track_overlay(
                        image_paths, accepted_tracks, raw_tracks,
                        unique_track_overlay_path,
                    )
                )
        except Exception:
            pass

        artifacts.extend(subprocess_result.get("countvid_artifacts", []))

        # Raw tracks manifest
        import json
        manifest_path = artifact_dir / "countvid_tracks.json"
        try:
            manifest_path.write_text(
                json.dumps(
                    {
                        "instance_count": instance_count,
                        "raw_track_count": len(raw_tracks),
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

        # Unique tracks manifest
        unique_manifest_path = artifact_dir / "countvid_unique_tracks.json"
        try:
            unique_manifest = {
                "unique_instance_count": instance_count,
                "raw_track_count": len(raw_tracks),
                "merged_track_count": len(raw_tracks) - instance_count,
                "unique_tracks": [],
            }
            for t in accepted_tracks:
                unique_manifest["unique_tracks"].append({
                    "track_id": t.get("track_id", ""),
                    "object": t.get("object", ""),
                    "frame_count": len(t.get("frame_indices", [])),
                    "member_seed_ids": t.get("member_seed_ids", []),
                })
            unique_manifest_path.write_text(
                json.dumps(unique_manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            artifacts.append(str(unique_manifest_path))
        except Exception:
            pass

        payload = {
            "instance_count": instance_count,
            "tracks": formatted_tracks,
            "frame_summaries": frame_summaries,
            "backend": backend_label,
            "pipeline_stats": {
                "frame_count": len(image_paths),
                "raw_instance_count": raw_instance_count,
                "raw_track_count": len(raw_tracks),
                "unique_track_count": instance_count,
                "merged_tracks": (len(raw_tracks) - instance_count) if apply_spatialscore_aggregation else 0,
                "dropped_tracks": raw_instance_count - len(raw_tracks) if raw_instance_count > len(raw_tracks) else 0,
                "point_threshold_px": point_threshold_px,
                "merge_overlap_min_frames": merge_overlap_min,
                "min_track_support": min_track_support,
                "apply_spatialscore_aggregation": apply_spatialscore_aggregation,
                **countvid_stats,
            },
            "backend_status": "ok",
            "artifact_descriptions": [
                {
                    "path": str(track_overlay_path),
                    "kind": "raw_track_overlay",
                    "description": "Raw propagated object tracks overlaid on sampled video frames.",
                },
                {
                    "path": str(unique_track_overlay_path),
                    "kind": "unique_track_overlay",
                    "description": "Final unique instances color-coded per instance across frames.",
                },
                {
                    "path": str(manifest_path),
                    "kind": "track_manifest",
                    "description": "Raw tracks manifest with instance count and frame summaries.",
                },
                {
                    "path": str(unique_manifest_path),
                    "kind": "unique_track_manifest",
                    "description": "Unique instance manifest showing which raw tracks were merged.",
                },
            ],
        }
        return self.success(payload=payload, artifacts=artifacts)
