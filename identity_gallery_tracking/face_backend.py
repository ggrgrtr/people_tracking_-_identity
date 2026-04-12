from pathlib import Path

import cv2
import numpy as np

from .reid import normalize_feature


def _resolve_optional_model(base_dir, explicit_path, candidate_names):
    candidates = []

    if explicit_path:
        explicit = Path(explicit_path)
        candidates.append(explicit)
        if not explicit.is_absolute():
            candidates.append(base_dir / explicit)
            candidates.append(base_dir / "weights" / explicit)

    for name in candidate_names:
        candidates.append(base_dir / name)
        candidates.append(base_dir / "weights" / name)

    seen = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    return None


class OptionalFaceBackend:
    def __init__(self, config, base_dir):
        self.config = config
        self.enabled = False
        self.description = "disabled (add YuNet + SFace models to enable face identity)"
        self.detector = None
        self.recognizer = None

        detector_cls = getattr(cv2, "FaceDetectorYN", None)
        recognizer_cls = getattr(cv2, "FaceRecognizerSF", None)
        if detector_cls is None or recognizer_cls is None:
            return


        # 2 модели
        detector_model = _resolve_optional_model(
            base_dir,
            config.face_detector_model,
            [
                "face_detection_yunet_2023mar.onnx",
                "face_detection_yunet_2022mar.onnx",
                "yunet.onnx",
            ],
        )
        recognizer_model = _resolve_optional_model(
            base_dir,
            config.face_recognizer_model,
            [
                "face_recognition_sface_2021dec.onnx",
                "face_recognition_sface_2022dec.onnx",
                "sface.onnx",
            ],
        )

        if detector_model is None or recognizer_model is None:
            return

        try:
            # Face backend полностью опционален: отсутствие моделей не должно ломать основной pipeline.
            self.detector = detector_cls.create(
                str(detector_model),
                "",
                (320, 320),
                config.face_score_threshold,
                config.face_nms_threshold,
                5000,
            )
            self.recognizer = recognizer_cls.create(str(recognizer_model), "")
            self.enabled = True
            self.description = (
                f"enabled ({detector_model.name} + {recognizer_model.name})"
            )
        except Exception as exc:
            self.enabled = False
            self.detector = None
            self.recognizer = None
            self.description = f"disabled (failed to initialize face models: {exc})"

    # вырезка области человека по боксу
    def _crop_person(self, frame, bbox):
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = [int(v) for v in bbox]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _select_face(self, faces):
        if faces is None or len(faces) == 0:
            return None

        best_face = None
        best_score = -1.0
        for face in faces:
            width = max(1.0, float(face[2]))
            height = max(1.0, float(face[3]))
            # Выбираем не просто самое уверенное лицо, а лучшее по сочетанию confidence и размера.
            score = float(face[-1]) * width * height
            if score > best_score:
                best_score = score
                best_face = face
        return best_face

    def extract(self, frame, bbox):
        # включен ли бек
        if not self.enabled:
            return None

        # вырезаем человека по person bbox
        # ищем лицо внутри  crop
        # выравниваем лицо по опорным точкам
        # получаем embedding для identity matching
        crop = self._crop_person(frame, bbox)
        if crop is None:
            return None

        crop_h, crop_w = crop.shape[:2]
        if crop_w < 48 or crop_h < 48:
            return None

        try:
            self.detector.setInputSize((crop_w, crop_h))
            _, faces = self.detector.detect(crop)
        except Exception:
            return None

        face = self._select_face(faces)
        if face is None:
            return None

        try:
            # alignCrop для выравнивания face embedding по стагдарту
            aligned_face = self.recognizer.alignCrop(crop, face)
            # берем эмбендинг
            feature = self.recognizer.feature(aligned_face)
        except Exception:
            return None

        # создаем вектор признаков и нормируем его для последующего сравнения косинусной метрикой
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        return normalize_feature(feature)

    def extract_batch(self, frame, boxes, candidate_indices=None):
        if not boxes:
            return []
        if not self.enabled:
            return [None] * len(boxes)

        selected_indices = None
        if candidate_indices is not None:
            selected_indices = {
                int(index)
                for index in candidate_indices
                if 0 <= int(index) < len(boxes)
            }

        features = [None] * len(boxes)
        for index, bbox in enumerate(boxes):
            #  признаки лица считаем только для выбранных кандидатов
            # чтобы не тратить время на каждую рамку сцены
            if selected_indices is not None and index not in selected_indices:
                continue
            features[index] = self.extract(frame, bbox)
        return features
