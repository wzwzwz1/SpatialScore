from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import math


@dataclass(frozen=True)
class View3D:
    view_id: str
    object: str
    frame_index: int
    bbox: Tuple[float, float, float, float]
    center_3d: Tuple[float, float, float]
    score: float | None = None
    track_id: str | None = None


@dataclass
class _Cluster:
    points: List["_ClusterPoint"]

    @property
    def frames(self) -> Set[int]:
        frames: Set[int] = set()
        for point in self.points:
            frames.update(point.frames)
        return frames


@dataclass(frozen=True)
class _ClusterPoint:
    point_id: str
    center_3d: Tuple[float, float, float]
    frames: Set[int]
    views: Tuple[View3D, ...]


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def constrained_greedy_cluster(
    views: Sequence[Dict[str, Any]] | Sequence[View3D],
    distance_threshold: float,
) -> List[Dict[str, Any]]:
    """Cluster 3D object views into physical instances using ViSRA CG.

    This follows Algorithm 2 in ViSRA Appendix C.1: if views carry ``track_id``,
    each track is pre-merged into a single point whose center is the mean of its
    member centers. Candidate point pairs are then sorted by 3D-center distance
    and greedily merged if the distance is below epsilon and their frame sets are
    disjoint.
    """
    normalized = [_as_view3d(v) for v in views]
    if not normalized:
        return []

    points = _build_initial_points(normalized)
    clusters: List[_Cluster] = [_Cluster([point]) for point in points]
    pairs: List[Tuple[float, int, int]] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            pairs.append((euclidean_distance(points[i].center_3d, points[j].center_3d), i, j))
    pairs.sort(key=lambda item: item[0])

    def find_cluster_index(point: _ClusterPoint) -> int:
        for index, cluster in enumerate(clusters):
            if any(item.point_id == point.point_id for item in cluster.points):
                return index
        raise ValueError(f"Point not found in any cluster: {point.point_id}")

    for distance, i, j in pairs:
        if distance > distance_threshold:
            break
        ci = find_cluster_index(points[i])
        cj = find_cluster_index(points[j])
        if ci == cj:
            continue
        frames_i = clusters[ci].frames
        frames_j = clusters[cj].frames
        if frames_i & frames_j:
            continue
        merged = _Cluster(clusters[ci].points + clusters[cj].points)
        for index in sorted([ci, cj], reverse=True):
            clusters.pop(index)
        clusters.append(merged)

    return [_cluster_to_instance(index, cluster) for index, cluster in enumerate(clusters)]


def _build_initial_points(views: Sequence[View3D]) -> List[_ClusterPoint]:
    grouped: Dict[str, List[View3D]] = {}
    untracked: List[View3D] = []
    for view in views:
        if view.track_id:
            grouped.setdefault(view.track_id, []).append(view)
        else:
            untracked.append(view)

    points: List[_ClusterPoint] = []
    for track_id, track_views in sorted(grouped.items()):
        points.append(_views_to_cluster_point(f"track:{track_id}", track_views))
    for view in untracked:
        points.append(_views_to_cluster_point(f"view:{view.view_id}", [view]))
    return points


def _views_to_cluster_point(point_id: str, views: Sequence[View3D]) -> _ClusterPoint:
    center = tuple(
        sum(values) / len(views)
        for values in zip(*(view.center_3d for view in views))
    )
    return _ClusterPoint(
        point_id=point_id,
        center_3d=tuple(float(value) for value in center),
        frames={view.frame_index for view in views},
        views=tuple(views),
    )


def _as_view3d(view: Dict[str, Any] | View3D) -> View3D:
    if isinstance(view, View3D):
        return view
    center = view.get("center_3d") or view.get("point_3d")
    if center is None or len(center) < 3:
        raise ValueError("3D view requires center_3d with three values")
    bbox = view.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    return View3D(
        view_id=str(view.get("view_id") or f"view_{view.get('frame_index', 0)}_{id(view)}"),
        object=str(view.get("object") or view.get("label") or "object"),
        frame_index=int(view.get("frame_index", 0)),
        bbox=tuple(float(x) for x in bbox[:4]),
        center_3d=tuple(float(x) for x in center[:3]),
        score=float(view["score"]) if isinstance(view.get("score"), (int, float)) else None,
        track_id=str(view.get("track_id")) if view.get("track_id") is not None else None,
    )


def _cluster_to_instance(index: int, cluster: _Cluster) -> Dict[str, Any]:
    original_views = [view for point in cluster.points for view in point.views]
    centers = [point.center_3d for point in original_views]
    center = [
        sum(values) / len(values)
        for values in zip(*centers)
    ]
    frames = sorted(point.frame_index for point in original_views)
    object_name = original_views[0].object
    xs = [value for point in original_views for value in (point.center_3d[0],)]
    ys = [value for point in original_views for value in (point.center_3d[1],)]
    zs = [value for point in original_views for value in (point.center_3d[2],)]
    return {
        "instance_id": f"{object_name}_{index + 1}",
        "object": object_name,
        "member_count": len(original_views),
        "frames": frames,
        "center_3d": center,
        "bbox_3d": {
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
        },
        "views": [
            {
                "view_id": point.view_id,
                "frame_index": point.frame_index,
                "bbox": list(point.bbox),
                "center_3d": list(point.center_3d),
                "score": point.score,
                "track_id": point.track_id,
            }
            for point in sorted(original_views, key=lambda item: (item.frame_index, item.view_id))
        ],
    }


def normalize_rex_bbox_predictions(outputs: Any, objects: Sequence[str], image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Best-effort parser for Rex-Omni bbox-style outputs.

    Rex-Omni wrappers are versioned independently; tests exercise the common
    extracted_predictions shape, while runtime falls back to a few simple list
    shapes so the tool fails softly rather than depending on one exact schema.
    """
    width, height = image_size
    first = outputs[0] if isinstance(outputs, list) and outputs else outputs
    predictions = first.get("extracted_predictions") if isinstance(first, dict) else None
    if predictions is None and isinstance(first, dict):
        predictions = first
    detections: List[Dict[str, Any]] = []
    if not isinstance(predictions, dict):
        return detections
    for object_name in objects:
        raw_items = predictions.get(object_name) or predictions.get(object_name.lower()) or []
        if isinstance(raw_items, dict):
            raw_items = raw_items.get("boxes") or raw_items.get("bboxes") or raw_items.get("detections") or []
        for item in raw_items:
            parsed = _parse_detection_item(item, width, height)
            if parsed is None:
                continue
            parsed["object"] = object_name
            detections.append(parsed)
    return detections


def _parse_detection_item(item: Any, width: int, height: int) -> Dict[str, Any] | None:
    if isinstance(item, dict):
        bbox = item.get("bbox") or item.get("box") or item.get("coords")
        if bbox is None and item.get("type") == "box":
            bbox = item.get("coords")
        if bbox is None or len(bbox) < 4:
            return None
        score = item.get("score") or item.get("confidence")
    elif isinstance(item, (list, tuple)) and len(item) >= 4:
        bbox = item[:4]
        score = item[4] if len(item) > 4 else None
    else:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return {
        "bbox": [x1, y1, x2, y2],
        "score": float(score) if isinstance(score, (int, float)) else None,
    }


def sample_bbox_pixels(
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    stride: int,
) -> List[Tuple[int, int]]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    ix1 = max(0, min(width - 1, int(round(x1))))
    iy1 = max(0, min(height - 1, int(round(y1))))
    ix2 = max(0, min(width - 1, int(round(x2))))
    iy2 = max(0, min(height - 1, int(round(y2))))
    if ix2 < ix1 or iy2 < iy1:
        return []
    step = max(1, int(stride))
    pixels = []
    for y in range(iy1, iy2 + 1, step):
        for x in range(ix1, ix2 + 1, step):
            pixels.append((x, y))
    if not pixels:
        pixels.append(((ix1 + ix2) // 2, (iy1 + iy2) // 2))
    return pixels


def median_point(points: Sequence[Sequence[float]]) -> List[float] | None:
    if not points:
        return None
    sorted_axes = []
    for axis in range(3):
        values = sorted(float(point[axis]) for point in points)
        mid = len(values) // 2
        if len(values) % 2:
            sorted_axes.append(values[mid])
        else:
            sorted_axes.append((values[mid - 1] + values[mid]) / 2.0)
    return sorted_axes
