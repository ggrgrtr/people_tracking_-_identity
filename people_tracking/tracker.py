from collections import deque

import cv2
import numpy as np

from .assignment import UNMATCHED_COST, hungarian
from .reid import blend_feature, blend_histogram, color_similarity, cosine_similarity
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
    nearest_frame_edge,
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


class Track:
    def __init__(self, track_id, bbox, feature, color_hist, frame_shape, config):
        self.id = track_id
        self.config = config
        self.min_confirmed_hits = config.min_confirmed_hits
        self.kalman = self._create_kalman(bbox)

        self.predicted_bbox = clip_bbox(bbox, frame_shape)
        self.smooth_bbox = clip_bbox(bbox, frame_shape)
        self.path = deque([get_center(self.smooth_bbox)], maxlen=config.path_len)

        self.feature = feature.astype(np.float32) if feature is not None else None
        self.color_hist = color_hist.astype(np.float32) if color_hist is not None else None
        self.identity_feature = self.feature.copy() if self.feature is not None else None
        self.identity_color_hist = self.color_hist.copy() if self.color_hist is not None else None
        self.shape_descriptor = _shape_descriptor_from_bbox(self.smooth_bbox, frame_shape)
        self.identity_shape_descriptor = self.shape_descriptor.copy()
        self.feature_bank = deque(maxlen=config.feature_bank_size)
        self.color_bank = deque(maxlen=config.color_bank_size)
        self.shape_bank = deque(maxlen=config.shape_bank_size)

        if self.feature is not None:
            self.feature_bank.append(self.feature)
        if self.color_hist is not None:
            self.color_bank.append(self.color_hist)
        if self.shape_descriptor is not None:
            self.shape_bank.append(self.shape_descriptor.copy())

        self.age = 1
        self.hits = 1
        self.missed_detections = 0
        self.inactive_age = 0
        self.last_seen_edge = nearest_frame_edge(self.smooth_bbox, frame_shape)
        self.exit_edge = None

    def _create_kalman(self, bbox):
        kalman = cv2.KalmanFilter(8, 4)
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
        kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.015
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.22
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
        self.last_seen_edge = nearest_frame_edge(self.smooth_bbox, frame_shape)
        append_path(self.path, get_center(self.smooth_bbox), min_distance=3)

    def update(self, bbox, feature, color_hist, frame_shape):
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
        self.last_seen_edge = nearest_frame_edge(self.smooth_bbox, frame_shape)
        self.exit_edge = None
        append_path(self.path, get_center(self.smooth_bbox), min_distance=3)

        self.hits += 1
        self.missed_detections = 0
        self.inactive_age = 0
        self._update_appearance(feature, color_hist)
        self._update_shape(frame_shape)

    def _update_appearance(self, feature, color_hist):
        if feature is not None:
            feature = feature.astype(np.float32)
            identity_similarity = cosine_similarity(self.identity_feature, feature)
            recent_similarity = cosine_similarity(self.feature, feature)
            accept_feature = (
                self.identity_feature is None
                or self.hits <= self.min_confirmed_hits + 1
                or identity_similarity >= self.config.identity_update_min_similarity
                or recent_similarity >= self.config.identity_update_min_similarity + 0.06
            )

            if accept_feature:
                blend_momentum = 0.90 if identity_similarity < 0.55 else 0.84
                self.feature = blend_feature(self.feature, feature, momentum=blend_momentum)
                self.feature_bank.append(feature)

                if self.identity_feature is None:
                    self.identity_feature = feature.copy()
                elif (
                    identity_similarity >= 0.45
                    or self.hits <= self.min_confirmed_hits + 2
                ):
                    self.identity_feature = blend_feature(
                        self.identity_feature,
                        feature,
                        momentum=self.config.identity_feature_momentum,
                    )

        if color_hist is not None:
            color_hist = color_hist.astype(np.float32)
            identity_color_similarity = color_similarity(self.identity_color_hist, color_hist)
            recent_color_similarity = color_similarity(self.color_hist, color_hist)
            accept_color = (
                self.identity_color_hist is None
                or self.hits <= self.min_confirmed_hits + 1
                or identity_color_similarity >= self.config.identity_color_update_min_similarity
                or recent_color_similarity >= self.config.identity_color_update_min_similarity + 0.04
            )

            if accept_color:
                blend_momentum = 0.88 if identity_color_similarity < 0.28 else 0.80
                self.color_hist = blend_histogram(
                    self.color_hist,
                    color_hist,
                    momentum=blend_momentum,
                )
                self.color_bank.append(color_hist)

                if self.identity_color_hist is None:
                    self.identity_color_hist = color_hist.copy()
                elif (
                    identity_color_similarity >= 0.18
                    or self.hits <= self.min_confirmed_hits + 2
                ):
                    self.identity_color_hist = blend_histogram(
                        self.identity_color_hist,
                        color_hist,
                        momentum=self.config.identity_color_momentum,
                    )

    def _update_shape(self, frame_shape):
        new_shape = _shape_descriptor_from_bbox(self.smooth_bbox, frame_shape)
        shape_similarity = _shape_similarity(self.identity_shape_descriptor, new_shape)

        self.shape_descriptor = new_shape.astype(np.float32)
        self.shape_bank.append(self.shape_descriptor.copy())

        if self.identity_shape_descriptor is None:
            self.identity_shape_descriptor = self.shape_descriptor.copy()
        elif (
            shape_similarity >= 0.52
            or self.hits <= self.min_confirmed_hits + 2
        ):
            self.identity_shape_descriptor = _blend_shape(
                self.identity_shape_descriptor,
                self.shape_descriptor,
                momentum=self.config.identity_shape_momentum,
            )

    def best_feature_similarity(self, feature):
        if feature is None:
            return -1.0

        weighted_scores = []
        if self.identity_feature is not None:
            weighted_scores.append((cosine_similarity(self.identity_feature, feature), 2.8))
        if self.feature is not None:
            weighted_scores.append((cosine_similarity(self.feature, feature), 1.9))

        bank_similarities = [
            cosine_similarity(bank_feature, feature)
            for bank_feature in self.feature_bank
        ]
        bank_similarities = [value for value in bank_similarities if value > -0.5]
        if bank_similarities:
            bank_similarities.sort(reverse=True)
            top_k = bank_similarities[: max(1, self.config.identity_bank_topk)]
            weighted_scores.append((sum(top_k) / len(top_k), 1.4))

        if not weighted_scores:
            return -1.0

        score_sum = sum(score * weight for score, weight in weighted_scores)
        weight_sum = sum(weight for _, weight in weighted_scores)
        return score_sum / max(weight_sum, 1e-8)

    def best_color_similarity(self, color_hist):
        if color_hist is None:
            return -1.0

        weighted_scores = []
        if self.identity_color_hist is not None:
            weighted_scores.append((color_similarity(self.identity_color_hist, color_hist), 2.4))
        if self.color_hist is not None:
            weighted_scores.append((color_similarity(self.color_hist, color_hist), 1.5))

        bank_similarities = [
            color_similarity(bank_hist, color_hist)
            for bank_hist in self.color_bank
        ]
        bank_similarities = [value for value in bank_similarities if value > -0.5]
        if bank_similarities:
            bank_similarities.sort(reverse=True)
            top_k = bank_similarities[: max(1, self.config.identity_bank_topk)]
            weighted_scores.append((sum(top_k) / len(top_k), 1.1))

        if not weighted_scores:
            return -1.0

        score_sum = sum(score * weight for score, weight in weighted_scores)
        weight_sum = sum(weight for _, weight in weighted_scores)
        return score_sum / max(weight_sum, 1e-8)

    def best_shape_similarity(self, bbox, frame_shape):
        if bbox is None:
            return -1.0

        shape = _shape_descriptor_from_bbox(bbox, frame_shape)
        weighted_scores = []
        if self.identity_shape_descriptor is not None:
            weighted_scores.append((_shape_similarity(self.identity_shape_descriptor, shape), 2.2))
        if self.shape_descriptor is not None:
            weighted_scores.append((_shape_similarity(self.shape_descriptor, shape), 1.4))

        bank_similarities = [
            _shape_similarity(bank_shape, shape)
            for bank_shape in self.shape_bank
        ]
        bank_similarities = [value for value in bank_similarities if value > -0.5]
        if bank_similarities:
            bank_similarities.sort(reverse=True)
            top_k = bank_similarities[: max(1, self.config.identity_bank_topk)]
            weighted_scores.append((sum(top_k) / len(top_k), 1.0))

        if not weighted_scores:
            return -1.0

        score_sum = sum(score * weight for score, weight in weighted_scores)
        weight_sum = sum(weight for _, weight in weighted_scores)
        return score_sum / max(weight_sum, 1e-8)

    def mark_missed(self):
        self.missed_detections += 1

    def mark_inactive(self):
        self.exit_edge = self.last_seen_edge
        self.inactive_age = 0

    def age_inactive(self):
        self.inactive_age += 1

    def is_confirmed(self, min_hits):
        return self.hits >= min_hits

    def absorb(self, other):
        if other is None or other is self:
            return

        self.hits = max(self.hits, other.hits)
        self.age = max(self.age, other.age)
        self.missed_detections = min(self.missed_detections, other.missed_detections)
        self.inactive_age = min(self.inactive_age, other.inactive_age)

        if other.feature is not None or other.color_hist is not None:
            self._update_appearance(other.feature, other.color_hist)

        if other.identity_feature is not None:
            self.identity_feature = blend_feature(
                self.identity_feature,
                other.identity_feature,
                momentum=self.config.identity_feature_momentum,
            )
        if other.identity_color_hist is not None:
            self.identity_color_hist = blend_histogram(
                self.identity_color_hist,
                other.identity_color_hist,
                momentum=self.config.identity_color_momentum,
            )
        if other.identity_shape_descriptor is not None:
            self.identity_shape_descriptor = _blend_shape(
                self.identity_shape_descriptor,
                other.identity_shape_descriptor,
                momentum=self.config.identity_shape_momentum,
            )

        for feature in other.feature_bank:
            self.feature_bank.append(feature.astype(np.float32))

        for color_hist in other.color_bank:
            self.color_bank.append(color_hist.astype(np.float32))
        for shape_descriptor in other.shape_bank:
            self.shape_bank.append(shape_descriptor.astype(np.float32))

        for point in other.path:
            append_path(self.path, point, min_distance=2)


def _active_match_cost(track, det_box, feature, color_hist, config):
    ref_box = track.predicted_bbox
    iou = compute_iou(ref_box, det_box)
    dist = center_distance(ref_box, det_box)
    size_sim = size_similarity(ref_box, det_box)
    appearance = track.best_feature_similarity(feature)
    color_score = track.best_color_similarity(color_hist)

    allowed_dist = max(72.0, max(ref_box[2], ref_box[3]) * 1.55)
    motion_confident = iou >= 0.22 or dist <= allowed_dist * 0.26
    if iou < 0.02 and dist > allowed_dist and appearance < config.active_reid_threshold:
        return None

    if (
        track.is_confirmed(config.min_confirmed_hits)
        and not motion_confident
        and appearance >= 0.0
        and appearance < max(config.active_reid_threshold, 0.50)
        and max(color_score, 0.0) < 0.18
        and iou < 0.20
    ):
        return None

    if (
        appearance >= 0.0
        and appearance < 0.22
        and max(color_score, 0.0) < 0.10
        and iou < 0.10
        and dist > allowed_dist * 0.32
    ):
        return None

    if (
        track.is_confirmed(config.min_confirmed_hits)
        and appearance >= 0.0
        and appearance < config.active_reid_threshold - 0.04
        and not motion_confident
        and size_sim < 0.62
    ):
        return None

    score = (
        iou * 2.05
        + max(0.0, 1.0 - dist / allowed_dist) * 0.90
        + size_sim * 0.55
        + max(0.0, appearance) * 3.35
        + max(0.0, color_score) * 0.28
    )

    if track.is_confirmed(config.min_confirmed_hits):
        score += max(0.0, appearance) * 0.75
        score += max(0.0, color_score) * 0.06
        if not motion_confident and appearance < config.active_reid_threshold:
            score -= 0.65

    return 10.0 - score


def _inactive_match_cost(track, det_box, feature, color_hist, frame_shape, config):
    if feature is None:
        return None

    appearance = track.best_feature_similarity(feature)
    color_score = track.best_color_similarity(color_hist)
    shape_score = track.best_shape_similarity(det_box, frame_shape)

    required_feature = config.inactive_reid_threshold
    required_color = config.min_inactive_color_similarity
    required_shape = config.min_inactive_shape_similarity
    if track.hits < config.inactive_min_reuse_hits:
        required_feature = max(required_feature, config.inactive_confident_feature_similarity)
        required_color = max(required_color, config.inactive_confident_color_similarity)
        required_shape = max(required_shape, config.inactive_confident_shape_similarity)
    if track.inactive_age >= config.inactive_long_gap_frames:
        required_feature += max(config.inactive_long_gap_extra_feature, 0.05)
        required_color += 0.02

    strong_reid = appearance >= config.inactive_confident_feature_similarity - 0.02
    appearance_shape_ok = (
        appearance >= required_feature - 0.01
        and shape_score >= required_shape
    )
    appearance_color_ok = (
        appearance >= required_feature
        and color_score >= required_color
    )

    if (
        appearance < required_feature
        and not strong_reid
        and not appearance_shape_ok
        and not appearance_color_ok
    ):
        return None

    if (
        track.hits < config.inactive_min_reuse_hits
        and (
            appearance < config.inactive_confident_feature_similarity + 0.03
            or shape_score < config.inactive_confident_shape_similarity
        )
    ):
        return None

    if (
        appearance < max(0.56, required_feature - 0.02)
        and color_score < max(config.inactive_confident_color_similarity, required_color)
        and shape_score < max(required_shape, 0.62)
    ):
        return None

    frame_diag = max(1.0, np.hypot(frame_shape[1], frame_shape[0]))
    dist = center_distance(track.smooth_bbox, det_box)
    size_sim = size_similarity(track.smooth_bbox, det_box)
    age_penalty = track.inactive_age / max(1, config.max_inactive_age)
    entry_edge = nearest_frame_edge(det_box, frame_shape)
    edge_bonus = 0.18 if track.exit_edge is not None and entry_edge == track.exit_edge else 0.0
    edge_penalty = 0.10 if track.exit_edge is not None and entry_edge != track.exit_edge else 0.0

    score = (
        max(0.0, appearance) * 3.95
        + max(0.0, color_score) * 0.55
        + max(0.0, shape_score) * 0.68
        + size_sim * 0.18
        + edge_bonus
        - edge_penalty
        - (dist / frame_diag) * 0.30
        - age_penalty * 0.26
    )

    if score < max(1.90, config.inactive_match_min_score):
        return None

    return 10.0 - score


def _archived_match_cost(track, det_box, feature, color_hist, frame_shape, config):
    if feature is None:
        return None

    appearance = track.best_feature_similarity(feature)
    color_score = track.best_color_similarity(color_hist)
    shape_score = track.best_shape_similarity(det_box, frame_shape)

    required_feature = config.archive_reid_threshold
    required_color = config.archive_color_threshold
    required_shape = config.archive_shape_threshold
    if track.hits < config.archive_min_reuse_hits:
        required_feature = max(required_feature, config.archive_reid_threshold + 0.03)
        required_color = max(required_color, config.archive_color_threshold + 0.03)
        required_shape = max(required_shape, config.archive_shape_threshold + 0.04)

    strong_reid = appearance >= required_feature + 0.05
    appearance_shape_ok = (
        appearance >= required_feature - 0.01
        and shape_score >= required_shape
    )
    appearance_color_ok = (
        appearance >= required_feature
        and color_score >= required_color
    )

    if (
        appearance < required_feature
        and not strong_reid
        and not appearance_shape_ok
        and not appearance_color_ok
    ):
        return None

    if (
        track.hits < config.archive_min_reuse_hits
        and (
            appearance < required_feature + 0.02
            or shape_score < required_shape
        )
    ):
        return None

    if (
        appearance < max(0.58, required_feature - 0.02)
        and color_score < required_color + 0.02
        and shape_score < required_shape
    ):
        return None

    frame_diag = max(1.0, np.hypot(frame_shape[1], frame_shape[0]))
    dist = center_distance(track.smooth_bbox, det_box)
    size_sim = size_similarity(track.smooth_bbox, det_box)
    age_penalty = track.inactive_age / max(1, config.max_archive_age)
    entry_edge = nearest_frame_edge(det_box, frame_shape)
    edge_bonus = 0.10 if track.exit_edge is not None and entry_edge == track.exit_edge else 0.0
    edge_penalty = 0.08 if track.exit_edge is not None and entry_edge != track.exit_edge else 0.0

    score = (
        max(0.0, appearance) * 4.35
        + max(0.0, color_score) * 0.48
        + max(0.0, shape_score) * 0.62
        + size_sim * 0.15
        + edge_bonus
        - edge_penalty
        - (dist / frame_diag) * 0.18
        - age_penalty * 0.12
    )

    if score < max(2.10, config.archive_match_min_score):
        return None

    return 10.0 - score


def _associate_tracks(
    tracks,
    detections,
    features,
    color_histograms,
    frame_shape,
    config,
    match_mode,
    return_cost_matrix=False,
):
    if not tracks or not detections:
        empty_cost = np.full((len(tracks), len(detections)), UNMATCHED_COST, dtype=np.float64)
        if return_cost_matrix:
            return [], list(range(len(tracks))), list(range(len(detections))), empty_cost
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost_matrix = np.full((len(tracks), len(detections)), UNMATCHED_COST, dtype=np.float64)

    for track_index, track in enumerate(tracks):
        for det_index, det_box in enumerate(detections):
            if not box_valid(det_box, frame_shape, config.min_box_w, config.min_box_h):
                continue

            if match_mode == "inactive":
                cost = _inactive_match_cost(
                    track,
                    det_box,
                    features[det_index],
                    color_histograms[det_index],
                    frame_shape,
                    config,
                )
            elif match_mode == "archive":
                cost = _archived_match_cost(
                    track,
                    det_box,
                    features[det_index],
                    color_histograms[det_index],
                    frame_shape,
                    config,
                )
            else:
                cost = _active_match_cost(
                    track,
                    det_box,
                    features[det_index],
                    color_histograms[det_index],
                    config,
                )

            if cost is not None:
                cost_matrix[track_index, det_index] = cost

    assignments = hungarian(cost_matrix)
    matches = []
    matched_tracks = set()
    matched_detections = set()

    for track_index, det_index in assignments:
        if cost_matrix[track_index, det_index] >= UNMATCHED_COST:
            continue

        matches.append((track_index, det_index))
        matched_tracks.add(track_index)
        matched_detections.add(det_index)

    unmatched_tracks = [index for index in range(len(tracks)) if index not in matched_tracks]
    unmatched_detections = [
        index for index in range(len(detections)) if index not in matched_detections
    ]

    if return_cost_matrix:
        return matches, unmatched_tracks, unmatched_detections, cost_matrix

    return matches, unmatched_tracks, unmatched_detections


class MultiObjectTracker:
    def __init__(self, config):
        self.config = config
        self.active_tracks = []
        self.inactive_tracks = []
        self.archived_tracks = []
        self.next_id = 1

    def _age_archived_tracks(self):
        alive_tracks = []
        for track in self.archived_tracks:
            track.age_inactive()
            if track.inactive_age <= self.config.max_archive_age:
                alive_tracks.append(track)
        self.archived_tracks = alive_tracks

    def _archive_track(self, track):
        if track is None:
            return

        for index, archived in enumerate(self.archived_tracks):
            if archived.id == track.id:
                archived.absorb(track)
                archived.predicted_bbox = track.predicted_bbox
                archived.smooth_bbox = track.smooth_bbox
                archived.last_seen_edge = track.last_seen_edge
                archived.exit_edge = track.exit_edge
                archived.inactive_age = track.inactive_age
                self.archived_tracks[index] = archived
                return

        self.archived_tracks.append(track)

    def _age_inactive_tracks(self):
        alive_tracks = []
        for track in self.inactive_tracks:
            track.age_inactive()
            if track.inactive_age <= self.config.max_inactive_age:
                alive_tracks.append(track)
            elif (
                track.is_confirmed(self.config.min_confirmed_hits)
                and track.hits >= self.config.archive_min_hits
            ):
                self._archive_track(track)
        self.inactive_tracks = alive_tracks

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
        if track.is_confirmed(self.config.min_confirmed_hits):
            score += 4.0

        score += min(track.hits, 12) * 1.5
        score += min(len(track.path), self.config.path_len) * 0.05
        score += min(4.0, bbox_area(track.smooth_bbox) / 7000.0)
        score -= track.missed_detections * 2.5
        return score

    def _tracks_are_duplicate(self, primary, secondary):
        if not self._boxes_look_duplicate(primary.smooth_bbox, secondary.smooth_bbox):
            return False

        feature_similarity = -1.0
        color_similarity_score = -1.0

        if secondary.feature is not None:
            feature_similarity = max(
                feature_similarity,
                primary.best_feature_similarity(secondary.feature),
            )
        if primary.feature is not None:
            feature_similarity = max(
                feature_similarity,
                secondary.best_feature_similarity(primary.feature),
            )

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

        if feature_similarity >= 0.64 or color_similarity_score >= 0.35:
            return True

        newer_duplicate = secondary.hits <= self.config.min_confirmed_hits + 1
        smaller_duplicate = bbox_area(secondary.smooth_bbox) <= bbox_area(primary.smooth_bbox) * 0.85
        return newer_duplicate and smaller_duplicate

    def _should_skip_new_track(self, det_box, feature, color_hist):
        det_center = get_center(det_box)
        for track in self.active_tracks:
            iou = compute_iou(track.smooth_bbox, det_box)
            nested_overlap = intersection_over_smaller(track.smooth_bbox, det_box)
            appearance = track.best_feature_similarity(feature)
            color_score = track.best_color_similarity(color_hist)

            if iou >= self.config.track_duplicate_iou:
                return True

            if (
                nested_overlap >= self.config.track_nested_overlap
                and point_in_bbox(det_center, track.smooth_bbox)
            ):
                return True

            if (
                nested_overlap >= 0.72
                and appearance >= max(self.config.active_reid_threshold, 0.60)
            ):
                return True

            if (
                point_in_bbox(det_center, track.smooth_bbox)
                and appearance >= 0.78
                and color_score >= 0.15
            ):
                return True

        return False

    def _deduplicate_active_tracks(self):
        if len(self.active_tracks) <= 1:
            return

        ordered_tracks = sorted(
            self.active_tracks,
            key=self._track_priority,
            reverse=True,
        )
        deduplicated = []

        for candidate in ordered_tracks:
            merged = False
            for kept in deduplicated:
                if self._tracks_are_duplicate(kept, candidate):
                    kept.absorb(candidate)
                    merged = True
                    break

            if not merged:
                deduplicated.append(candidate)

        self.active_tracks = deduplicated

    def _is_confident_inactive_match(
        self,
        track,
        det_box,
        feature,
        color_hist,
        frame_shape,
        assigned_cost,
        row_margin,
        col_margin,
    ):
        if feature is None:
            return False

        appearance = track.best_feature_similarity(feature)
        color_score = track.best_color_similarity(color_hist)
        shape_score = track.best_shape_similarity(det_box, frame_shape)
        raw_score = 10.0 - assigned_cost

        strong_identity = (
            appearance >= max(self.config.inactive_confident_feature_similarity, self.config.inactive_reid_threshold + 0.07)
            or (
                appearance >= self.config.inactive_reid_threshold + 0.03
                and shape_score >= self.config.inactive_confident_shape_similarity
            )
            or (
                appearance >= self.config.inactive_reid_threshold
                and shape_score >= self.config.inactive_confident_shape_similarity + 0.02
                and color_score >= self.config.min_inactive_color_similarity + 0.04
            )
        )

        if raw_score < max(2.00, self.config.inactive_match_min_score):
            return False

        if (
            track.hits < self.config.inactive_min_reuse_hits
            and (
                appearance < self.config.inactive_confident_feature_similarity + 0.03
                or shape_score < self.config.inactive_confident_shape_similarity
            )
        ):
            return False

        if (
            track.inactive_age >= self.config.inactive_long_gap_frames
            and appearance < self.config.inactive_reid_threshold + 0.03
            and not strong_identity
        ):
            return False

        if (
            row_margin < self.config.inactive_match_row_margin
            or col_margin < self.config.inactive_match_col_margin
        ) and appearance < self.config.inactive_reid_threshold + 0.06 and not strong_identity:
            return False

        return True

    def _is_confident_archive_match(
        self,
        track,
        det_box,
        feature,
        color_hist,
        frame_shape,
        assigned_cost,
        row_margin,
        col_margin,
    ):
        if feature is None:
            return False

        appearance = track.best_feature_similarity(feature)
        color_score = track.best_color_similarity(color_hist)
        shape_score = track.best_shape_similarity(det_box, frame_shape)
        raw_score = 10.0 - assigned_cost

        strong_identity = (
            appearance >= self.config.archive_reid_threshold + 0.05
            or (
                appearance >= self.config.archive_reid_threshold + 0.02
                and shape_score >= self.config.archive_shape_threshold
            )
            or (
                appearance >= self.config.archive_reid_threshold
                and color_score >= self.config.archive_color_threshold + 0.05
                and shape_score >= self.config.archive_shape_threshold + 0.04
            )
        )

        if raw_score < max(2.10, self.config.archive_match_min_score):
            return False

        if (
            track.hits < self.config.archive_min_reuse_hits
            and (
                appearance < self.config.archive_reid_threshold + 0.05
                or shape_score < self.config.archive_shape_threshold + 0.04
            )
        ):
            return False

        if (
            appearance < self.config.archive_reid_threshold - 0.01
            and (
                color_score < self.config.archive_color_threshold + 0.02
                or shape_score < self.config.archive_shape_threshold
            )
        ):
            return False

        if (
            row_margin < self.config.archive_match_row_margin
            or col_margin < self.config.archive_match_col_margin
        ):
            return (
                appearance >= self.config.archive_reid_threshold + 0.04
                and shape_score >= self.config.archive_shape_threshold
            )

        return strong_identity or (
            appearance >= self.config.archive_reid_threshold + 0.01
            and color_score >= self.config.archive_color_threshold
            and shape_score >= self.config.archive_shape_threshold
        )

    def _filter_inactive_matches(
        self,
        inactive_tracks,
        detections,
        features,
        color_histograms,
        frame_shape,
        matches,
        cost_matrix,
    ):
        if not matches:
            return []

        confident_matches = []
        for track_index, det_index in matches:
            assigned_cost = cost_matrix[track_index, det_index]
            row_costs = [
                cost_matrix[track_index, idx]
                for idx in range(cost_matrix.shape[1])
                if idx != det_index and cost_matrix[track_index, idx] < UNMATCHED_COST
            ]
            col_costs = [
                cost_matrix[idx, det_index]
                for idx in range(cost_matrix.shape[0])
                if idx != track_index and cost_matrix[idx, det_index] < UNMATCHED_COST
            ]

            second_row_cost = min(row_costs) if row_costs else UNMATCHED_COST
            second_col_cost = min(col_costs) if col_costs else UNMATCHED_COST
            row_margin = second_row_cost - assigned_cost
            col_margin = second_col_cost - assigned_cost

            track = inactive_tracks[track_index]
            if self._is_confident_inactive_match(
                track,
                detections[det_index],
                features[det_index],
                color_histograms[det_index],
                frame_shape,
                assigned_cost,
                row_margin,
                col_margin,
            ):
                confident_matches.append((track_index, det_index))

        return confident_matches

    def _filter_archive_matches(
        self,
        archived_tracks,
        detections,
        features,
        color_histograms,
        frame_shape,
        matches,
        cost_matrix,
    ):
        if not matches:
            return []

        confident_matches = []
        for track_index, det_index in matches:
            assigned_cost = cost_matrix[track_index, det_index]
            row_costs = [
                cost_matrix[track_index, idx]
                for idx in range(cost_matrix.shape[1])
                if idx != det_index and cost_matrix[track_index, idx] < UNMATCHED_COST
            ]
            col_costs = [
                cost_matrix[idx, det_index]
                for idx in range(cost_matrix.shape[0])
                if idx != track_index and cost_matrix[idx, det_index] < UNMATCHED_COST
            ]

            second_row_cost = min(row_costs) if row_costs else UNMATCHED_COST
            second_col_cost = min(col_costs) if col_costs else UNMATCHED_COST
            row_margin = second_row_cost - assigned_cost
            col_margin = second_col_cost - assigned_cost

            track = archived_tracks[track_index]
            if self._is_confident_archive_match(
                track,
                detections[det_index],
                features[det_index],
                color_histograms[det_index],
                frame_shape,
                assigned_cost,
                row_margin,
                col_margin,
            ):
                confident_matches.append((track_index, det_index))

        return confident_matches

    def predict_only(self, frame_shape):
        self._age_archived_tracks()
        self._age_inactive_tracks()
        for track in self.active_tracks:
            track.predict(frame_shape)
        self._deduplicate_active_tracks()

    def update(self, detections, features, color_histograms, frame_shape):
        self._age_archived_tracks()
        self._age_inactive_tracks()
        for track in self.active_tracks:
            track.predict(frame_shape)

        matches, _, unmatched_detections = _associate_tracks(
            self.active_tracks,
            detections,
            features,
            color_histograms,
            frame_shape,
            self.config,
            match_mode="active",
        )

        matched_active = set()
        for track_index, det_index in matches:
            self.active_tracks[track_index].update(
                detections[det_index],
                features[det_index],
                color_histograms[det_index],
                frame_shape,
            )
            matched_active.add(track_index)

        still_active = []
        newly_inactive = []
        for track_index, track in enumerate(self.active_tracks):
            if track_index not in matched_active:
                track.mark_missed()

            if track.missed_detections > self.config.max_missed_detections:
                track.mark_inactive()
                newly_inactive.append(track)
            else:
                still_active.append(track)

        self.active_tracks = still_active
        self.inactive_tracks.extend(newly_inactive)

        remaining_detection_indices = unmatched_detections[:]
        if self.inactive_tracks and remaining_detection_indices:
            remaining_boxes = [detections[index] for index in remaining_detection_indices]
            remaining_features = [features[index] for index in remaining_detection_indices]
            remaining_colors = [color_histograms[index] for index in remaining_detection_indices]

            inactive_matches, _, unmatched_after_inactive, inactive_cost_matrix = _associate_tracks(
                self.inactive_tracks,
                remaining_boxes,
                remaining_features,
                remaining_colors,
                frame_shape,
                self.config,
                match_mode="inactive",
                return_cost_matrix=True,
            )
            inactive_matches = self._filter_inactive_matches(
                self.inactive_tracks,
                remaining_boxes,
                remaining_features,
                remaining_colors,
                frame_shape,
                inactive_matches,
                inactive_cost_matrix,
            )

            revived_inactive = set()
            revived_local_detections = set()
            for inactive_index, local_det_index in inactive_matches:
                det_index = remaining_detection_indices[local_det_index]
                track = self.inactive_tracks[inactive_index]
                track.update(
                    detections[det_index],
                    features[det_index],
                    color_histograms[det_index],
                    frame_shape,
                )
                track.missed_detections = 0
                track.inactive_age = 0
                self.active_tracks.append(track)
                revived_inactive.add(inactive_index)
                revived_local_detections.add(local_det_index)

            self.inactive_tracks = [
                track
                for index, track in enumerate(self.inactive_tracks)
                if index not in revived_inactive
            ]
            remaining_detection_indices = [
                remaining_detection_indices[index]
                for index in range(len(remaining_detection_indices))
                if index not in revived_local_detections
            ]

        if self.archived_tracks and remaining_detection_indices:
            remaining_boxes = [detections[index] for index in remaining_detection_indices]
            remaining_features = [features[index] for index in remaining_detection_indices]
            remaining_colors = [color_histograms[index] for index in remaining_detection_indices]

            archive_matches, _, _, archive_cost_matrix = _associate_tracks(
                self.archived_tracks,
                remaining_boxes,
                remaining_features,
                remaining_colors,
                frame_shape,
                self.config,
                match_mode="archive",
                return_cost_matrix=True,
            )
            archive_matches = self._filter_archive_matches(
                self.archived_tracks,
                remaining_boxes,
                remaining_features,
                remaining_colors,
                frame_shape,
                archive_matches,
                archive_cost_matrix,
            )

            revived_archived = set()
            revived_archive_detections = set()
            for archived_index, local_det_index in archive_matches:
                det_index = remaining_detection_indices[local_det_index]
                track = self.archived_tracks[archived_index]
                track.update(
                    detections[det_index],
                    features[det_index],
                    color_histograms[det_index],
                    frame_shape,
                )
                track.missed_detections = 0
                track.inactive_age = 0
                self.active_tracks.append(track)
                revived_archived.add(archived_index)
                revived_archive_detections.add(local_det_index)

            self.archived_tracks = [
                track
                for index, track in enumerate(self.archived_tracks)
                if index not in revived_archived
            ]
            remaining_detection_indices = [
                remaining_detection_indices[index]
                for index in range(len(remaining_detection_indices))
                if index not in revived_archive_detections
            ]

        for det_index in remaining_detection_indices:
            if self._should_skip_new_track(
                detections[det_index],
                features[det_index],
                color_histograms[det_index],
            ):
                continue

            new_track = Track(
                self.next_id,
                detections[det_index],
                features[det_index],
                color_histograms[det_index],
                frame_shape,
                self.config,
            )
            self.active_tracks.append(new_track)
            self.next_id += 1

        self._deduplicate_active_tracks()

    def visible_tracks(self):
        return [
            track
            for track in self.active_tracks
            if track.is_confirmed(self.config.min_confirmed_hits)
            or track.age <= self.config.yolo_interval + 1
        ]
