import json
import math
import time
from collections import deque

import cv2
import numpy as np

from .reid import (
    blend_feature,
    blend_histogram,
    color_similarity,
    cosine_similarity,
)
from .utils import get_center


def _shape_descriptor_from_bbox(bbox, frame_shape):
    # Shape descriptor - это маленький вектор с геометрией человека в кадре.
    # Он не годится как самостоятельная идентификация,
    # но хорошо помогает отсеивать явно неподходящих кандидатов.
    frame_h, frame_w = frame_shape[:2]
    _, _, w, h = [int(v) for v in bbox]
    safe_w = max(1.0, float(frame_w))
    safe_h = max(1.0, float(frame_h))
    safe_box_h = max(1.0, float(h))
    return np.array(
        [
            float(w) / safe_w,
            float(h) / safe_h,
            float(w) / safe_box_h,
        ],
        dtype=np.float32,
    )


def _shape_similarity(shape_a, shape_b):
    if shape_a is None or shape_b is None:
        return -1.0

    width_sim = min(shape_a[0], shape_b[0]) / max(shape_a[0], shape_b[0], 1e-6)
    height_sim = min(shape_a[1], shape_b[1]) / max(shape_a[1], shape_b[1], 1e-6)
    aspect_sim = min(shape_a[2], shape_b[2]) / max(shape_a[2], shape_b[2], 1e-6)
    return 0.30 * width_sim + 0.45 * height_sim + 0.25 * aspect_sim


def _blend_shape(old_shape, new_shape, momentum=0.92):
    if old_shape is None:
        return new_shape.astype(np.float32)
    if new_shape is None:
        return old_shape.astype(np.float32)
    return (momentum * old_shape + (1.0 - momentum) * new_shape).astype(np.float32)


class IdentityRecord:
    def __init__(self, identity_id, tracklet, frame_id, elapsed_seconds, frame_shape, config):
        self.id = identity_id
        self.config = config
        # В identity кладем уже не "сырое одно наблюдение", а стартовую галерею признаков,
        # которая дальше будет усредняться и пополняться банками примеров.
        self.feature = tracklet.identity_feature.copy() if tracklet.identity_feature is not None else None
        self.color_hist = (
            tracklet.identity_color_hist.copy() if tracklet.identity_color_hist is not None else None
        )
        self.face_feature = (
            tracklet.identity_face_feature.copy() if tracklet.identity_face_feature is not None else None
        )
        self.shape_descriptor = _shape_descriptor_from_bbox(tracklet.smooth_bbox, frame_shape)

        self.feature_bank = deque(maxlen=config.identity_bank_size)
        self.color_bank = deque(maxlen=config.identity_bank_size)
        self.face_bank = deque(maxlen=max(6, config.identity_bank_size // 3))
        self.shape_bank = deque(maxlen=config.identity_bank_size)

        if self.feature is not None:
            self.feature_bank.append(self.feature)
        if self.color_hist is not None:
            self.color_bank.append(self.color_hist)
        if self.face_feature is not None:
            self.face_bank.append(self.face_feature)
        self.shape_bank.append(self.shape_descriptor.copy())

        self.first_seen_frame = frame_id
        self.first_seen_time_sec = round(elapsed_seconds, 2)
        self.last_seen_frame = frame_id
        self.last_seen_time_sec = round(elapsed_seconds, 2)
        self.observations = 0
        self.tracklets_count = 0
        self.distance_px = 0.0
        self.trajectory = []
        self.record_observation(tracklet, frame_id, elapsed_seconds, frame_shape)

    def record_observation(
        self,
        tracklet,
        frame_id,
        elapsed_seconds,
        frame_shape,
        update_gallery=True,
        observed=True,
    ):
        if not observed:
            return

        self.last_seen_frame = frame_id
        self.last_seen_time_sec = round(elapsed_seconds, 2)
        self.observations += 1

        if update_gallery and tracklet.identity_feature is not None:
            self.feature = blend_feature(
                self.feature,
                tracklet.identity_feature,
                momentum=self.config.identity_feature_momentum,
            )
            self.feature_bank.append(tracklet.identity_feature.astype(np.float32))

        if update_gallery and tracklet.identity_color_hist is not None:
            self.color_hist = blend_histogram(
                self.color_hist,
                tracklet.identity_color_hist,
                momentum=self.config.identity_color_momentum,
            )
            self.color_bank.append(tracklet.identity_color_hist.astype(np.float32))

        if update_gallery and tracklet.identity_face_feature is not None:
            self.face_feature = blend_feature(
                self.face_feature,
                tracklet.identity_face_feature,
                momentum=0.88,
            )
            self.face_bank.append(tracklet.identity_face_feature.astype(np.float32))

        shape = _shape_descriptor_from_bbox(tracklet.smooth_bbox, frame_shape)
        if update_gallery:
            self.shape_descriptor = _blend_shape(
                self.shape_descriptor,
                shape,
                momentum=self.config.identity_shape_momentum,
            )
            self.shape_bank.append(shape.astype(np.float32))

        center = get_center(tracklet.smooth_bbox)
        bbox = [int(value) for value in tracklet.smooth_bbox]
        should_append = True
        if self.trajectory:
            prev_center = self.trajectory[-1]["center"]
            distance = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
            if distance < self.config.route_log_min_distance:
                should_append = False
            else:
                self.distance_px += distance

        if should_append:
            self.trajectory.append(
                {
                    "frame": frame_id,
                    "time_sec": round(elapsed_seconds, 2),
                    "center": [int(center[0]), int(center[1])],
                    "bbox": bbox,
                }
            )

    def mark_tracklet_finished(self):
        self.tracklets_count += 1

    def best_feature_similarity(self, feature):
        if feature is None:
            return -1.0

        scores = []
        if self.feature is not None:
            scores.append((cosine_similarity(self.feature, feature), 2.8))
        bank_scores = [cosine_similarity(value, feature) for value in self.feature_bank]
        bank_scores = [value for value in bank_scores if value > -0.5]
        if bank_scores:
            bank_scores.sort(reverse=True)
            scores.append((sum(bank_scores[:4]) / min(len(bank_scores), 4), 1.2))

        if not scores:
            return -1.0
        score_sum = sum(score * weight for score, weight in scores)
        weight_sum = sum(weight for _, weight in scores)
        return score_sum / max(weight_sum, 1e-8)

    def best_color_similarity(self, color_hist):
        if color_hist is None:
            return -1.0

        scores = []
        if self.color_hist is not None:
            scores.append((color_similarity(self.color_hist, color_hist), 2.0))
        bank_scores = [color_similarity(value, color_hist) for value in self.color_bank]
        bank_scores = [value for value in bank_scores if value > -0.5]
        if bank_scores:
            bank_scores.sort(reverse=True)
            scores.append((sum(bank_scores[:4]) / min(len(bank_scores), 4), 1.0))

        if not scores:
            return -1.0
        score_sum = sum(score * weight for score, weight in scores)
        weight_sum = sum(weight for _, weight in scores)
        return score_sum / max(weight_sum, 1e-8)

    def best_face_similarity(self, face_feature):
        if face_feature is None:
            return -1.0

        scores = []
        if self.face_feature is not None:
            scores.append((cosine_similarity(self.face_feature, face_feature), 3.2))
        bank_scores = [cosine_similarity(value, face_feature) for value in self.face_bank]
        bank_scores = [value for value in bank_scores if value > -0.5]
        if bank_scores:
            bank_scores.sort(reverse=True)
            scores.append((sum(bank_scores[:2]) / min(len(bank_scores), 2), 1.1))

        if not scores:
            return -1.0
        score_sum = sum(score * weight for score, weight in scores)
        weight_sum = sum(weight for _, weight in scores)
        return score_sum / max(weight_sum, 1e-8)

    def best_shape_similarity(self, bbox, frame_shape):
        shape = _shape_descriptor_from_bbox(bbox, frame_shape)
        scores = []
        if self.shape_descriptor is not None:
            scores.append((_shape_similarity(self.shape_descriptor, shape), 2.0))
        bank_scores = [_shape_similarity(value, shape) for value in self.shape_bank]
        bank_scores = [value for value in bank_scores if value > -0.5]
        if bank_scores:
            bank_scores.sort(reverse=True)
            scores.append((sum(bank_scores[:4]) / min(len(bank_scores), 4), 1.0))

        score_sum = sum(score * weight for score, weight in scores)
        weight_sum = sum(weight for _, weight in scores)
        return score_sum / max(weight_sum, 1e-8)


class IdentityManager:
    def __init__(self, config):
        self.config = config
        self.identities = {}
        self.next_identity_id = 1
        self.last_frame_shape = None

    def _match_score(self, identity, tracklet, frame_shape, frame_id):
        # Учебная идея:
        # у identity нет одного "магического" признака.
        # Итоговая уверенность собирается из нескольких сигналов:
        # лицо, внешний вид, цвет, форма и временная давность последнего наблюдения.
        face_score = identity.best_face_similarity(tracklet.identity_face_feature)
        appearance = identity.best_feature_similarity(tracklet.identity_feature)
        color_score = identity.best_color_similarity(tracklet.identity_color_hist)
        shape_score = identity.best_shape_similarity(tracklet.smooth_bbox, frame_shape)
        age_penalty = max(0.0, frame_id - identity.last_seen_frame) / max(1, self.config.identity_max_age)

        if (
            tracklet.identity_face_feature is not None
            and identity.face_feature is not None
            and face_score >= self.config.identity_face_threshold
        ):
            # Если лицо совпало достаточно уверенно, appearance/color/shape становятся только уточняющими сигналами.
            return (
                face_score * 5.4
                + max(0.0, appearance) * 1.3
                + max(0.0, color_score) * 0.20
                + max(0.0, shape_score) * 0.20
                - age_penalty * 0.08
            )

        if appearance < self.config.identity_reid_threshold:
            return None

        if (
            appearance < self.config.identity_reid_threshold + 0.03
            and color_score < self.config.identity_color_threshold
            and shape_score < self.config.identity_shape_threshold
        ):
            return None

        # Без лица appearance становится главным сигналом,
        # а color и shape помогают не перепутать похожих по embedding кандидатов.
        return (
            max(0.0, appearance) * 4.8
            + max(0.0, color_score) * 0.55
            + max(0.0, shape_score) * 0.55
            - age_penalty * 0.10
        )

    def _create_identity(self, tracklet, frame_id, elapsed_seconds, frame_shape):
        identity_id = self.next_identity_id
        self.next_identity_id += 1
        self.identities[identity_id] = IdentityRecord(
            identity_id,
            tracklet,
            frame_id,
            elapsed_seconds,
            frame_shape,
            self.config,
        )
        return identity_id

    def _match_identity(self, tracklet, occupied_identity_ids, frame_shape, frame_id):
        # Здесь формируется список всех identity, которые вообще имеют право претендовать на tracklet.
        # Затем выбирается лучший кандидат, но только если он:
        # 1. достаточно уверенный сам по себе;
        # 2. заметно лучше второго по качеству кандидата.
        candidates = []
        for identity_id, identity in self.identities.items():
            if identity_id in occupied_identity_ids:
                continue
            score = self._match_score(identity, tracklet, frame_shape, frame_id)
            if score is not None:
                candidates.append((score, identity_id))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        best_score, best_identity_id = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else -1.0
        if best_score < 3.10:
            return None
        # Если два кандидата слишком близки, безопаснее не матчинить и создать новую identity.
        if second_score > 0.0 and best_score - second_score < self.config.identity_match_margin:
            return None
        return best_identity_id

    def observe_tracklets(
        self,
        tracklets,
        frame_id,
        elapsed_seconds,
        frame_shape,
        update_gallery=True,
    ):
        self.last_frame_shape = frame_shape
        occupied_identity_ids = {track.person_id for track in tracklets if track.person_id is not None}

        for track in tracklets:
            if not track.is_confirmed():
                continue
            # Prediction-only кадры не должны искусственно продлевать жизнь identity в журнале наблюдений.
            if not track.was_observed():
                continue

            created_identity = False
            if track.person_id is None and track.identity_ready():
                # Identity присваивается только зрелым tracklet:
                # они уже пережили несколько кадров и накопили достаточно признаков.
                identity_id = self._match_identity(
                    track,
                    occupied_identity_ids,
                    frame_shape,
                    frame_id,
                )
                if identity_id is None:
                    identity_id = self._create_identity(track, frame_id, elapsed_seconds, frame_shape)
                    created_identity = True
                track.person_id = identity_id
                occupied_identity_ids.add(identity_id)

            if track.person_id is None:
                continue
            if created_identity:
                continue

            identity = self.identities.get(track.person_id)
            if identity is None:
                continue
            identity.record_observation(
                track,
                frame_id,
                elapsed_seconds,
                frame_shape,
                update_gallery=update_gallery,
                observed=track.was_observed(),
            )

    def finalize_tracklets(self, tracklets):
        for track in tracklets:
            if track.person_id is None:
                continue
            identity = self.identities.get(track.person_id)
            if identity is not None:
                identity.mark_tracklet_finished()

    def _route_image_name(self, identity_id):
        return f"identity_{int(identity_id):03d}_route.png"

    def _draw_route_image(self, identity):
        frame_shape = self.last_frame_shape or (
            self.config.camera_height,
            self.config.camera_width,
            3,
        )
        height, width = frame_shape[:2]
        canvas = np.full((height, width, 3), 28, dtype=np.uint8)

        if identity.trajectory:
            points = [tuple(point["center"]) for point in identity.trajectory]
            total_segments = max(1, len(points) - 1)
            for index in range(1, len(points)):
                progress = index / total_segments
                shade = int(20 + progress * 215)
                color = (shade, shade, shade)
                thickness = 2 if progress < 0.5 else 3
                cv2.line(canvas, points[index - 1], points[index], color, thickness)

            cv2.circle(canvas, points[0], 6, (0, 255, 0), -1)
            cv2.circle(canvas, points[-1], 6, (0, 0, 255), -1)
            if len(points) >= 2:
                cv2.arrowedLine(canvas, points[-2], points[-1], (245, 245, 245), 3, tipLength=0.25)

        cv2.putText(
            canvas,
            f"identity_id: {identity.id}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            canvas,
            "Identity route: darker -> lighter",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 128, 255),
            1,
        )
        return canvas

    def save(self, output_path, routes_dir, session_duration=None):
        routes_dir.mkdir(parents=True, exist_ok=True)
        identities_payload = []

        for identity_id in sorted(self.identities):
            identity = self.identities[identity_id]
            route_image_path = routes_dir / self._route_image_name(identity_id)
            route_image = self._draw_route_image(identity)
            cv2.imwrite(str(route_image_path), route_image)

            identities_payload.append(
                {
                    "identity_id": identity.id,
                    "first_seen_frame": identity.first_seen_frame,
                    "first_seen_time_sec": identity.first_seen_time_sec,
                    "last_seen_frame": identity.last_seen_frame,
                    "last_seen_time_sec": identity.last_seen_time_sec,
                    "observations": identity.observations,
                    "tracklets_count": identity.tracklets_count,
                    "distance_px": round(identity.distance_px, 2),
                    "trajectory_points": len(identity.trajectory),
                    "has_face_gallery": bool(identity.face_bank),
                    "has_reid_gallery": bool(identity.feature_bank),
                    "route_image": str(route_image_path),
                }
            )

        payload = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project": "identity_gallery_tracking",
            "identities": identities_payload,
        }
        if session_duration is not None:
            payload["session_duration_sec"] = round(session_duration, 2)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
