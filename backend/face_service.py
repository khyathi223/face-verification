from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import (
    CROPS_DIR,
    DETECTION_CONFIDENCE_THRESHOLD,
    MAX_IMAGE_SIZE,
    MATCH_THRESHOLD,
    PROCESSED_DIR,
)


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray
    crop_path: Path | None = None
    status: str | None = None
    matched_user_id: str | None = None
    matched_name: str | None = None
    score: float | None = None


class FaceService:
    def __init__(self) -> None:
        self.mode = "opencv-fallback"
        self.app = None
        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=-1, det_size=(MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
            self.mode = "insightface-arcface"
        except Exception as exc:
            print(f"InsightFace unavailable; using OpenCV fallback detector. Reason: {exc}")

    def decode_image(self, content: bytes) -> np.ndarray:
        arr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not read image. Please upload a valid JPG, JPEG, or PNG file.")
        return image

    def resize_image(self, image: np.ndarray, max_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(max_size / max(height, width), 1.0)
        if scale == 1.0:
            return image
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        embedding = embedding.astype("float32")
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def detect_faces(self, image: np.ndarray, save_crops: bool = True) -> list[DetectedFace]:
        image = self.resize_image(image)
        if self.app is not None:
            return self._detect_with_insightface(image, save_crops)
        return self._detect_with_opencv(image, save_crops)

    def _detect_with_insightface(self, image: np.ndarray, save_crops: bool) -> list[DetectedFace]:
        faces = []
        for face in self.app.get(image):
            confidence = float(face.det_score)
            if confidence < DETECTION_CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_path = self._save_crop(image, (x1, y1, x2, y2)) if save_crops else None
            faces.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    embedding=self.normalize_embedding(face.embedding),
                    crop_path=crop_path,
                )
            )
        return faces

    def _detect_with_opencv(self, image: np.ndarray, save_crops: bool) -> list[DetectedFace]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        raw_faces = self.haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
        )
        faces = []
        for (x, y, w, h) in raw_faces:
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            crop_path = self._save_crop(image, (x1, y1, x2, y2)) if save_crops else None
            faces.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    confidence=0.85,
                    embedding=self._fallback_embedding(image[y1:y2, x1:x2]),
                    crop_path=crop_path,
                )
            )
        return faces

    def _fallback_embedding(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray).astype("float32") / 255.0
        hist = cv2.calcHist([resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype("float32")
        vector = np.concatenate([gray.flatten(), hist])
        return self.normalize_embedding(vector)

    def _save_crop(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> Path:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        crop_path = CROPS_DIR / f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(crop_path), crop)
        return crop_path

    def draw_faces(self, image: np.ndarray, faces: list[DetectedFace]) -> Path:
        output = self.resize_image(image).copy()
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            color = (40, 180, 90) if face.status == "Verified User" else (40, 90, 230)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = face.status or f"Face {face.confidence:.2f}"
            if face.matched_name:
                label = f"{face.matched_name} ({face.score:.2f})"
            cv2.putText(
                output,
                label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        path = PROCESSED_DIR / f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(path), output)
        return path

    def compare(self, query: np.ndarray, candidates: list[tuple[str, str, np.ndarray]]):
        best = None
        for user_id, name, stored in candidates:
            if stored.shape != query.shape:
                continue
            score = float(np.dot(query, stored))
            if best is None or score > best[2]:
                best = (user_id, name, score)
        if best and best[2] >= MATCH_THRESHOLD:
            return best
        return None


face_service = FaceService()

