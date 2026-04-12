from collections import deque

import cv2
import numpy as np

from .assignment import UNMATCHED_COST, hungarian
from .reid import (
    blend_feature,
    blend_histogram,
    color_similarity,
    cosine_similarity,
)
from .utils import (
    append_path,
    bbox_area,
    bbox_from_center,
    box_valid,
    center_distance,
    clip_bbox,
    compute_iou,
    get_center,
    intersection_over_smaller,
    point_in_bbox,
    size_similarity,
    smooth_bbox,
)


def _shape_descriptor_from_bbox(bbox, frame_shape):
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


# мат фильтр Калмана

# Tracklet - это короткий трек, который живет в течение нескольких кадров
# отвечает за поддержание непрерывности наблюдения за объектом

#  не детектит людей сам и не выдает финальную долговременную identity
# Его зона ответственности: создать tracklet, вести его между кадрами
# сопоставлять detection с уже существующими треками
# обновлять признаки и удалять потерянные или дублирующиеся треки
class Tracklet:
    def __init__(
        self,
        tracklet_id,
        bbox,
        feature,
        color_hist,
        face_feature,
        frame_shape,
        config,
    ):
        self.id = tracklet_id
        self.person_id = None
        self.config = config
        self.min_confirmed_hits = config.min_confirmed_hits
        self.kalman = self._create_kalman(bbox)

        self.predicted_bbox = clip_bbox(bbox, frame_shape)
        # сглаживание рамки
        self.smooth_bbox = clip_bbox(bbox, frame_shape)
        self.path = deque([get_center(self.smooth_bbox)], maxlen=config.path_len)

        self.feature = feature.astype(np.float32) if feature is not None else None
        self.color_hist = color_hist.astype(np.float32) if color_hist is not None else None
        self.face_feature = face_feature.astype(np.float32) if face_feature is not None else None
        # identity признаки это эмбендинги личностей, хранящиеся в долговременной памяти 
        self.identity_feature = self.feature.copy() if self.feature is not None else None
        self.identity_color_hist = self.color_hist.copy() if self.color_hist is not None else None
        self.identity_face_feature = self.face_feature.copy() if self.face_feature is not None else None
        self.shape_descriptor = _shape_descriptor_from_bbox(self.smooth_bbox, frame_shape)
        self.identity_shape_descriptor = self.shape_descriptor.copy()

        self.feature_bank = deque(maxlen=10)
        self.color_bank = deque(maxlen=10)
        self.face_bank = deque(maxlen=6)
        self.shape_bank = deque(maxlen=10)

        if self.feature is not None:
            self.feature_bank.append(self.feature)
        if self.color_hist is not None:
            self.color_bank.append(self.color_hist)
        if self.face_feature is not None:
            self.face_bank.append(self.face_feature)
        self.shape_bank.append(self.shape_descriptor.copy())

        self.age = 1
        self.hits = 1
        self.missed_detections = 0
        self.feature_updates = 1 if self.feature is not None else 0
        self.face_updates = 1 if self.face_feature is not None else 0
        self.seen_frames = 1
        self.observed_in_current_frame = True

    def _create_kalman(self, bbox):
        # состояние Kalman здесь имеет вид
        # [cx, cy, w, h, vx, vy, vw, vh]
        #  первые 4 значения - положение и размер bbox
        # а последние 4 - их скорости

        # Kalman использует эту структуру для прогнозирования будущего положения человека
        kalman = cv2.KalmanFilter(8, 4)
        #  как состояние меняется во времени
        kalman.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        # реально измеряем только центр и размер рамки
        # скорости напрямую не наблюдаем, Kalman выводит их сам из последовательности кадров
        kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        # processNoiseCov отвечает за то, насколько модель движения может сомневаться в своем прогнозе
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.015
        # measurementNoiseCov отвечает за доверие к новому измерению от детектора
        #  насколько шумные измерения
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.22
        # начальная неопределенность
        kalman.errorCovPost = np.eye(8, dtype=np.float32)

        cx, cy = get_center(bbox)
        w = max(self.config.min_box_w, int(bbox[2]))
        h = max(self.config.min_box_h, int(bbox[3]))
        kalman.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
        return kalman

    def _state_to_bbox(self, state, frame_shape):
        cx, cy, w, h = [float(value) for value in state[:4]]
        w = max(self.config.min_box_w, int(abs(w)))
        h = max(self.config.min_box_h, int(abs(h)))
        return clip_bbox(bbox_from_center((int(cx), int(cy)), w, h), frame_shape)

    def predict(self, frame_shape):
        prediction = self.kalman.predict().reshape(-1)
        self.age += 1
        # Prediction продолжает жизнь tracklet между detect-pass, но не считается новым наблюдением
        self.predicted_bbox = self._state_to_bbox(prediction, frame_shape)
        self.smooth_bbox = smooth_bbox(
            self.smooth_bbox,
            self.predicted_bbox,
            frame_shape,
            self.config.prediction_center_alpha,
            self.config.prediction_size_alpha,
            self.config.size_jump_ratio,
            self.config.min_box_w,
            self.config.min_box_h,
            self.config.center_deadzone_px,
            self.config.size_deadzone_px,
        )
        append_path(self.path, get_center(self.smooth_bbox), min_distance=3)

    def update(self, bbox, feature, color_hist, face_feature, frame_shape):
        bbox = clip_bbox(bbox, frame_shape)
        cx, cy = get_center(bbox)
        measurement = np.array(
            [[cx], [cy], [int(bbox[2])], [int(bbox[3])]],
            dtype=np.float32,
        )
        self.kalman.correct(measurement)

        self.predicted_bbox = bbox
        self.smooth_bbox = smooth_bbox(
            self.smooth_bbox,
            bbox,
            frame_shape,
            self.config.measurement_center_alpha,
            self.config.measurement_size_alpha,
            self.config.size_jump_ratio,
            self.config.min_box_w,
            self.config.min_box_h,
            self.config.center_deadzone_px,
            self.config.size_deadzone_px,
        )
        append_path(self.path, get_center(self.smooth_bbox), min_distance=3)

        self.hits += 1
        self.seen_frames += 1
        self.missed_detections = 0
        # Только после реального измерения из детектора tracklet считается наблюденным в текущем кадре.
        self.observed_in_current_frame = True
        self._update_appearance(feature, color_hist, face_feature)
        self._update_shape(frame_shape)

    def mark_missed(self):
        self.missed_detections += 1
        self.observed_in_current_frame = False

    def is_confirmed(self, min_hits=None):
        min_hits = self.min_confirmed_hits if min_hits is None else min_hits
        return self.hits >= min_hits

    def was_observed(self):
        return self.observed_in_current_frame

    def _update_appearance(self, feature, color_hist, face_feature):
        if feature is not None:
            feature = feature.astype(np.float32)
            self.feature = blend_feature(self.feature, feature, momentum=0.86)
            self.feature_bank.append(feature)
            self.feature_updates += 1
            if self.identity_feature is None:
                self.identity_feature = feature.copy()
            else:
                # Identity-признак обновляется более инерционно, чем локальный tracklet feature.
                self.identity_feature = blend_feature(
                    self.identity_feature,
                    feature,
                    momentum=0.92,
                )

        if color_hist is not None:
            color_hist = color_hist.astype(np.float32)
            self.color_hist = blend_histogram(self.color_hist, color_hist, momentum=0.84)
            self.color_bank.append(color_hist)
            if self.identity_color_hist is None:
                self.identity_color_hist = color_hist.copy()
            else:
                self.identity_color_hist = blend_histogram(
                    self.identity_color_hist,
                    color_hist,
                    momentum=0.90,
                )

        if face_feature is not None:
            face_feature = face_feature.astype(np.float32)
            self.face_feature = blend_feature(self.face_feature, face_feature, momentum=0.80)
            self.face_bank.append(face_feature)
            self.face_updates += 1
            if self.identity_face_feature is None:
                self.identity_face_feature = face_feature.copy()
            else:
                self.identity_face_feature = blend_feature(
                    self.identity_face_feature,
                    face_feature,
                    momentum=0.88,
                )

    def _update_shape(self, frame_shape):
        new_shape = _shape_descriptor_from_bbox(self.smooth_bbox, frame_shape)
        self.shape_descriptor = new_shape.astype(np.float32)
        self.shape_bank.append(self.shape_descriptor.copy())
        self.identity_shape_descriptor = _blend_shape(
            self.identity_shape_descriptor,
            self.shape_descriptor,
            momentum=0.92,
        )

    def best_feature_similarity(self, feature):
        if feature is None:
            return -1.0

        scores = []
        if self.identity_feature is not None:
            scores.append((cosine_similarity(self.identity_feature, feature), 2.4))
        if self.feature is not None:
            scores.append((cosine_similarity(self.feature, feature), 1.8))
        bank_scores = [cosine_similarity(value, feature) for value in self.feature_bank]
        bank_scores = [value for value in bank_scores if value > -0.5]
        if bank_scores:
            bank_scores.sort(reverse=True)
            scores.append((sum(bank_scores[:3]) / min(len(bank_scores), 3), 1.1))

        if not scores:
            return -1.0
        score_sum = sum(score * weight for score, weight in scores)
        weight_sum = sum(weight for _, weight in scores)
        return score_sum / max(weight_sum, 1e-8)

    def best_color_similarity(self, color_hist):
        if color_hist is None:
            return -1.0

        scores = []
        if self.identity_color_hist is not None:
            scores.append((color_similarity(self.identity_color_hist, color_hist), 2.0))
        if self.color_hist is not None:
            scores.append((color_similarity(self.color_hist, color_hist), 1.4))
        bank_scores = [color_similarity(value, color_hist) for value in self.color_bank]
        bank_scores = [value for value in bank_scores if value > -0.5]
        if bank_scores:
            bank_scores.sort(reverse=True)
            scores.append((sum(bank_scores[:3]) / min(len(bank_scores), 3), 1.0))

        if not scores:
            return -1.0
        score_sum = sum(score * weight for score, weight in scores)
        weight_sum = sum(weight for _, weight in scores)
        return score_sum / max(weight_sum, 1e-8)

    def best_face_similarity(self, face_feature):
        if face_feature is None:
            return -1.0

        scores = []
        if self.identity_face_feature is not None:
            scores.append((cosine_similarity(self.identity_face_feature, face_feature), 2.8))
        if self.face_feature is not None:
            scores.append((cosine_similarity(self.face_feature, face_feature), 1.9))
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
        if bbox is None:
            return -1.0

        shape = _shape_descriptor_from_bbox(bbox, frame_shape)
        scores = []
        if self.identity_shape_descriptor is not None:
            scores.append((_shape_similarity(self.identity_shape_descriptor, shape), 2.0))
        if self.shape_descriptor is not None:
            scores.append((_shape_similarity(self.shape_descriptor, shape), 1.3))

        if not scores:
            return -1.0
        score_sum = sum(score * weight for score, weight in scores)
        weight_sum = sum(weight for _, weight in scores)
        return score_sum / max(weight_sum, 1e-8)

    def identity_ready(self):
        return (
            self.is_confirmed(self.config.identity_min_hits)
            and self.feature_updates >= self.config.identity_min_feature_updates
            and bbox_area(self.smooth_bbox) >= self.config.min_identity_box_area
        )


def _active_match_cost(track, det_box, feature, color_hist, face_feature, config):
    # Учебная идея функции:
    # мы не пытаемся матчинить tracklet и detection по одному признаку.
    # Вместо этого собираем "стоимость несовпадения" из геометрии, движения и внешности.
    # Чем выше итоговый score, тем лучше совпадение. В конце он переводится в cost через 10 - score,
    # потому что Hungarian в этой реализации минимизирует стоимость.
    ref_box = track.predicted_bbox
    iou = compute_iou(ref_box, det_box)
    dist = center_distance(ref_box, det_box)
    size_sim = size_similarity(ref_box, det_box)
    appearance = track.best_feature_similarity(feature)
    color_score = track.best_color_similarity(color_hist)
    face_score = track.best_face_similarity(face_feature)

    allowed_dist = max(72.0, max(ref_box[2], ref_box[3]) * 1.55)
    motion_ok = iou >= 0.18 or dist <= allowed_dist * 0.30

    if face_feature is not None and face_score >= config.active_match_face_threshold:
        # Лицо - самый сильный сигнал, поэтому при хорошем face match оно доминирует в стоимости.
        score = (
            max(0.0, face_score) * 3.8
            + iou * 1.6
            + max(0.0, 1.0 - dist / allowed_dist) * 0.8
            + size_sim * 0.4
        )
        return 10.0 - score

    if iou < 0.02 and dist > allowed_dist and appearance < config.active_match_reid_threshold:
        return None

    if (
        track.is_confirmed()
        and not motion_ok
        and appearance >= 0.0
        and appearance < config.active_match_reid_threshold
        and iou < 0.18
    ):
        return None

    # Если лица нет или оно слабое, локальный матч строится на смеси:
    # 1. геометрии рамки,
    # 2. правдоподобия движения,
    # 3. similarity внешнего вида.
    score = (
        iou * 2.25
        + max(0.0, 1.0 - dist / allowed_dist) * 0.95
        + size_sim * 0.55
        + max(0.0, appearance) * 2.10
        + max(0.0, color_score) * 0.22
    )
    return 10.0 - score


# Ассоциация tracklet и detection - это ключевая часть трекера, которая влияет на его способность сохранять непрерывность треков и правильно определять объекты
def _associate_tracklets(tracklets, detections, features, color_histograms, face_features, frame_shape, config):
    if not tracklets or not detections:
        return [], list(range(len(tracklets))), list(range(len(detections)))

    # Hungarian assignment получает единую матрицу стоимости и выбирает глобально согласованное сопоставление
    # Это лучше жадного подхода, потому что жадный матч может "украсть" хорошую detection у более подходящего tracklet
    # для каждой пары “существующий tracklet + новая detection” считается цена
    cost_matrix = np.full((len(tracklets), len(detections)), UNMATCHED_COST, dtype=np.float64)

    for track_index, track in enumerate(tracklets):
        for det_index, det_box in enumerate(detections):
            if not box_valid(det_box, frame_shape, config.min_box_w, config.min_box_h):
                continue
            cost = _active_match_cost(
                track,
                det_box,
                features[det_index],
                color_histograms[det_index],
                face_features[det_index],
                config,
            )
            if cost is not None:
                cost_matrix[track_index, det_index] = cost

    assignments = hungarian(cost_matrix)
    matches = []
    matched_tracks = set()
    matched_detections = set()

    # Hungarian может вернуть пары даже для плохих ячеек.
    # Поэтому после него обязательно фильтруем по порогу UNMATCHED_COST.
    for track_index, det_index in assignments:
        if cost_matrix[track_index, det_index] >= UNMATCHED_COST:
            continue
        matches.append((track_index, det_index))
        matched_tracks.add(track_index)
        matched_detections.add(det_index)

    unmatched_tracks = [index for index in range(len(tracklets)) if index not in matched_tracks]
    unmatched_detections = [index for index in range(len(detections)) if index not in matched_detections]
    return matches, unmatched_tracks, unmatched_detections


class TrackletTracker:
    def __init__(self, config):
        self.config = config
        self.active_tracklets = []
        self.next_id = 1

    def _boxes_look_duplicate(self, primary_bbox, secondary_bbox):
        iou = compute_iou(primary_bbox, secondary_bbox)
        nested_overlap = intersection_over_smaller(primary_bbox, secondary_bbox)
        secondary_center_inside = point_in_bbox(get_center(secondary_bbox), primary_bbox)
        primary_center_inside = point_in_bbox(get_center(primary_bbox), secondary_bbox)

        if iou >= self.config.track_duplicate_iou:
            return True

        return (
            nested_overlap >= self.config.track_nested_overlap
            and (secondary_center_inside or primary_center_inside)
        )

    def _track_priority(self, track):
        score = 0.0
        if track.is_confirmed():
            score += 4.0
        score += min(track.hits, 12) * 1.3
        score += min(len(track.path), self.config.path_len) * 0.05
        score += min(4.0, bbox_area(track.smooth_bbox) / 7000.0)
        score -= track.missed_detections * 2.0
        return score

    def _tracks_are_duplicate(self, primary, secondary):
        if not self._boxes_look_duplicate(primary.smooth_bbox, secondary.smooth_bbox):
            return False

        feature_similarity = -1.0
        color_similarity_score = -1.0

        if secondary.feature is not None:
            feature_similarity = max(feature_similarity, primary.best_feature_similarity(secondary.feature))
        if primary.feature is not None:
            feature_similarity = max(feature_similarity, secondary.best_feature_similarity(primary.feature))

        if secondary.color_hist is not None:
            color_similarity_score = max(
                color_similarity_score,
                primary.best_color_similarity(secondary.color_hist),
            )
        if primary.color_hist is not None:
            color_similarity_score = max(
                color_similarity_score,
                secondary.best_color_similarity(primary.color_hist),
            )

        if feature_similarity >= 0.62 or color_similarity_score >= 0.34:
            return True

        newer_duplicate = secondary.hits <= self.config.min_confirmed_hits + 1
        smaller_duplicate = bbox_area(secondary.smooth_bbox) <= bbox_area(primary.smooth_bbox) * 0.85
        return newer_duplicate and smaller_duplicate

    def _deduplicate_active_tracklets(self):
        if len(self.active_tracklets) <= 1:
            return []

        ordered_tracklets = sorted(self.active_tracklets, key=self._track_priority, reverse=True)
        deduplicated = []
        removed = []
        for candidate in ordered_tracklets:
            merged = False
            for kept in deduplicated:
                if self._tracks_are_duplicate(kept, candidate):
                    # Сохраняем удаленный дубликат как finished tracklet, чтобы не потерять финализацию статистики.
                    merged = True
                    removed.append(candidate)
                    break
            if not merged:
                deduplicated.append(candidate)
        self.active_tracklets = deduplicated
        return removed

    def predict_only(self, frame_shape):
        for track in self.active_tracklets:
            track.predict(frame_shape)
        return self._deduplicate_active_tracklets()

    def reid_candidate_detection_indices(self, detections, frame_shape):
        if not detections:
            return []

        if not self.active_tracklets:
            return list(range(len(detections)))

        # Кандидаты для дорогого ReID - это новые, сомнительные или identity-критичные детекции.
        candidate_indices = set()
        for det_index, det_box in enumerate(detections):
            best_track = None
            best_iou = 0.0
            best_dist_ratio = float("inf")

            for track in self.active_tracklets:
                ref_box = track.predicted_bbox
                iou = compute_iou(ref_box, det_box)
                allowed_dist = max(72.0, max(ref_box[2], ref_box[3]) * 1.55)
                dist_ratio = center_distance(ref_box, det_box) / max(allowed_dist, 1e-6)

                if iou > best_iou or (abs(iou - best_iou) <= 1e-6 and dist_ratio < best_dist_ratio):
                    best_track = track
                    best_iou = iou
                    best_dist_ratio = dist_ratio

            if best_track is None:
                candidate_indices.add(det_index)
                continue

            if not best_track.is_confirmed() or best_track.person_id is None:
                candidate_indices.add(det_index)
                continue

            if best_iou < 0.10 and best_dist_ratio > 0.34:
                candidate_indices.add(det_index)
                continue

            if (
                best_iou < 0.18
                and best_dist_ratio > 0.24
                and bbox_area(det_box) >= self.config.min_identity_box_area
            ):
                candidate_indices.add(det_index)

        return sorted(candidate_indices)

    def update(self, detections, features, color_histograms, face_features, frame_shape):
        # Сначала все текущие tracklet живут шагом prediction.
        # Это делает обработку симметричной: каждый трек сначала прогнозируется вперед,
        # а затем, если нашлась подходящая detection, корректируется измерением.
        for track in self.active_tracklets:
            track.predict(frame_shape)

        matches, unmatched_tracks, unmatched_detections = _associate_tracklets(
            self.active_tracklets,
            detections,
            features,
            color_histograms,
            face_features,
            frame_shape,
            self.config,
        )

        for track_index, det_index in matches:
            self.active_tracklets[track_index].update(
                detections[det_index],
                features[det_index],
                color_histograms[det_index],
                face_features[det_index],
                frame_shape,
            )

        finished_tracklets = []
        still_active = []
        unmatched_track_set = set(unmatched_tracks)
        for track_index, track in enumerate(self.active_tracklets):
            if track_index in unmatched_track_set:
                track.mark_missed()

            if track.missed_detections > self.config.max_missed_detections:
                finished_tracklets.append(track)
            else:
                still_active.append(track)

        self.active_tracklets = still_active

        for det_index in unmatched_detections:
            new_track = Tracklet(
                self.next_id,
                detections[det_index],
                features[det_index],
                color_histograms[det_index],
                face_features[det_index],
                frame_shape,
                self.config,
            )
            self.active_tracklets.append(new_track)
            self.next_id += 1

        finished_tracklets.extend(self._deduplicate_active_tracklets())
        return finished_tracklets

    def finish_all(self):
        finished = list(self.active_tracklets)
        self.active_tracklets = []
        return finished

    def visible_tracklets(self):
        return [
            track
            for track in self.active_tracklets
            if track.is_confirmed()
            or track.age <= self.config.yolo_interval + 1
        ]
