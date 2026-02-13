from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    from mtcnn.mtcnn import MTCNN  # TensorFlow/Keras-based
except Exception as e:
    raise ImportError("Install mtcnn (TF/Keras): pip install mtcnn tensorflow") from e


Landmarks5 = np.ndarray  # shape (5, 2)


@dataclass
class DetectedFace:
    box: np.ndarray        # (4,) => [x1, y1, x2, y2]
    prob: float
    landmarks: Landmarks5  # (5, 2)


def _template_5pts(output_size: tuple[int, int] = (112, 112)) -> np.ndarray:
    ref = np.array(
        [
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left mouth
            [70.7299, 92.2041],  # right mouth
        ],
        dtype=np.float32,
    )
    w, h = output_size
    ref[:, 0] *= w / 112.0
    ref[:, 1] *= h / 112.0
    return ref


def estimate_affine_from_5pts(src_landmarks: Landmarks5, output_size: tuple[int, int] = (112, 112)) -> np.ndarray:
    if src_landmarks.shape != (5, 2):
        raise ValueError(f"Expected (5,2) landmarks, got {src_landmarks.shape}")

    dst = _template_5pts(output_size=output_size)
    M, _ = cv2.estimateAffinePartial2D(
        src_landmarks.astype(np.float32),
        dst.astype(np.float32),
        method=cv2.LMEDS,
    )
    if M is None:
        raise RuntimeError("Could not estimate affine transform from landmarks.")
    return M.astype(np.float32)


def warp_face(image_bgr: np.ndarray, M_2x3: np.ndarray, output_size: tuple[int, int] = (112, 112)) -> np.ndarray:
    w, h = output_size
    return cv2.warpAffine(image_bgr, M_2x3, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


class FaceDetectorMTCNN:
    """
    TensorFlow/Keras-based MTCNN face detector using `mtcnn` package.
    Returns 5 keypoints: left_eye, right_eye, nose, mouth_left, mouth_right.
    """

    def __init__(self) -> None:
        self.detector = MTCNN()

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """
        Returns list of DetectedFace objects.
        """
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected HxWx3 image (BGR)")

        # mtcnn expects RGB
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Each detection: {'box':[x,y,w,h], 'confidence':float, 'keypoints':{...}}
        dets = self.detector.detect_faces(rgb) or []

        faces: list[DetectedFace] = []
        for d in dets:
            conf = float(d.get("confidence", 0.0))
            x, y, w, h = d["box"]
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)

            kp = d.get("keypoints", {})
            # keypoints: left_eye, right_eye, nose, mouth_left, mouth_right
            lm = np.array(
                [
                    kp["left_eye"],
                    kp["right_eye"],
                    kp["nose"],
                    kp["mouth_left"],
                    kp["mouth_right"],
                ],
                dtype=np.float32,
            )

            faces.append(
                DetectedFace(
                    box=np.array([x1, y1, x2, y2], dtype=np.float32),
                    prob=conf,
                    landmarks=lm,
                )
            )

        faces.sort(key=lambda f: f.prob, reverse=True)
        return faces

    def detect_and_align(
        self,
        image_bgr: np.ndarray,
        output_size: tuple[int, int] = (112, 112),
        min_prob: float = 0.90,
        pick: str = "best",  # "best" or "largest"
    ) -> dict[str, Any]:
        faces = self.detect(image_bgr)
        if not faces:
            return {"faces": [], "selected": None}

        faces_f = [f for f in faces if f.prob >= min_prob] or faces

        if pick == "largest":

            def area(face: DetectedFace) -> float:
                x1, y1, x2, y2 = face.box
                return max(0.0, x2 - x1) * max(0.0, y2 - y1)

            selected = max(faces_f, key=area)
        else:
            selected = faces_f[0]

        M = estimate_affine_from_5pts(selected.landmarks, output_size=output_size)
        aligned = warp_face(image_bgr, M, output_size=output_size)

        return {
            "faces": faces,
            "selected": selected,
            "affine": M,
            "aligned_bgr": aligned,
        }