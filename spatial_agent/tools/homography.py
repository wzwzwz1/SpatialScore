from __future__ import annotations

from spatial_agent.tools.base import BaseSpatialTool


class EstimateHomographyMatrixTool(BaseSpatialTool):
    name = "EstimateHomographyMatrix"
    description = "Estimate a homography matrix between two scene views."
    args_schema = {
        "type": "object",
        "properties": {
            "image": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "num_keypoints": {"type": "integer"},
            "ratio_th": {"type": "number"},
            "ransac_reproj_threshold": {"type": "number"},
        },
        "required": ["image"],
    }
    returns_schema = {"type": "object"}

    def __init__(self, config) -> None:
        self.config = config

    def invoke(self, **kwargs):
        image_paths = kwargs.get("image") or kwargs.get("images") or []
        if len(image_paths) < 2:
            return self.error("EstimateHomographyMatrix requires at least two image paths.")

        try:  # pragma: no cover - optional runtime dependency
            import cv2
            import numpy as np
        except Exception:
            return self.unavailable(
                "Homography estimation requires OpenCV feature matching dependencies, which are not available in the current environment."
            )

        first = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
        second = cv2.imread(image_paths[1], cv2.IMREAD_GRAYSCALE)
        if first is None or second is None:
            return self.error("Failed to load one or more input images for homography estimation.")

        num_keypoints = int(kwargs.get("num_keypoints", 1200))
        ratio_th = float(kwargs.get("ratio_th", 0.75))
        ransac_reproj_threshold = float(kwargs.get("ransac_reproj_threshold", 5.0))

        orb = cv2.ORB_create(nfeatures=num_keypoints)
        keypoints_a, descriptors_a = orb.detectAndCompute(first, None)
        keypoints_b, descriptors_b = orb.detectAndCompute(second, None)
        if descriptors_a is None or descriptors_b is None:
            return self.error("Unable to extract feature descriptors from one or more images.")

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue
            first_match, second_match = pair
            if first_match.distance < ratio_th * second_match.distance:
                good_matches.append(first_match)

        if len(good_matches) < 4:
            return self.error("Not enough inlier candidates to estimate a homography matrix.")

        src = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_reproj_threshold)
        if matrix is None:
            return self.error("OpenCV failed to estimate a homography matrix.")

        inliers_count = int(mask.sum()) if mask is not None else 0
        return self.success(
            payload={
                "homography_matrix": matrix.tolist(),
                "inliers_count": inliers_count,
                "total_matches": len(good_matches),
                "backend": "opencv_orb_ransac",
            }
        )
