from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    # Параметры источника и отображения.
    camera_width: int = 640
    camera_height: int = 360
    mirror_camera: bool = True
    threaded_camera_capture: bool = True

    # Параметры детекции: как часто запускать YOLO и насколько агрессивно сжимать кадр.
    yolo_interval: int = 4
    empty_scene_yolo_interval: int = 7
    yolo_scale: float = 0.60
    yolo_imgsz: int = 288
    yolo_conf: float = 0.45

    # Минимальный размер бокса, при котором объект еще считается полезным.
    min_box_w: int = 28
    min_box_h: int = 56
    min_identity_box_area: int = 5200

    # Параметры жизненного цикла tracklet.
    path_len: int = 80
    max_missed_detections: int = 8
    min_confirmed_hits: int = 2

    # Сглаживание bbox после Kalman prediction и после реального измерения детектором.
    prediction_center_alpha: float = 0.16
    prediction_size_alpha: float = 0.08
    measurement_center_alpha: float = 0.40
    measurement_size_alpha: float = 0.18
    size_jump_ratio: float = 0.18
    center_deadzone_px: int = 6
    size_deadzone_px: int = 6

    # Пороги локального сопоставления detection <-> active tracklet и борьбы с дублями.
    active_match_reid_threshold: float = 0.42
    active_match_face_threshold: float = 0.38
    detector_duplicate_iou: float = 0.58
    detector_nested_overlap: float = 0.76
    track_duplicate_iou: float = 0.64
    track_nested_overlap: float = 0.82

    # Параметры долговременной identity-памяти.
    identity_min_hits: int = 4
    identity_min_feature_updates: int = 2
    identity_match_margin: float = 0.22
    identity_reid_threshold: float = 0.70
    identity_color_threshold: float = 0.24
    identity_shape_threshold: float = 0.62
    identity_face_threshold: float = 0.36
    identity_max_age: int = 7200
    identity_bank_size: int = 18
    identity_feature_momentum: float = 0.95
    identity_color_momentum: float = 0.92
    identity_shape_momentum: float = 0.94

    # Параметры ReID-модели и color descriptor.
    reid_backbone: str = "resnet50_gem"
    reid_weights: str = "reid_resnet50_msmt17.pth"
    reid_input_height: int = 224
    reid_input_width: int = 112
    reid_padding: float = 0.08
    reid_interval: int = 2
    reid_force_count: int = 2
    max_reid_detections: int = 3
    reid_num_parts: int = 3
    reid_gem_p: float = 3.0
    reid_global_weight: float = 1.0
    reid_part_weight: float = 0.65
    reid_hist_h_bins: int = 12
    reid_hist_s_bins: int = 8

    # Параметры optional face backend.
    face_detector_model: str = ""
    face_recognizer_model: str = ""
    face_score_threshold: float = 0.88
    face_nms_threshold: float = 0.30

    # Параметры сохранения маршрутов и выходного видео.
    route_log_min_distance: int = 4
    writer_default_fps: float = 25.0
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "identity_tracking_output"
    )
