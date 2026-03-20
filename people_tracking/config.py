from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    camera_width: int = 640
    camera_height: int = 360
    mirror_camera: bool = True

    yolo_interval: int = 3
    empty_scene_yolo_interval: int = 6
    yolo_scale: float = 0.65
    yolo_imgsz: int = 320
    yolo_conf: float = 0.45

    min_box_w: int = 28
    min_box_h: int = 56

    path_len: int = 80
    max_missed_detections: int = 8
    max_inactive_age: int = 240
    min_confirmed_hits: int = 2

    active_reid_threshold: float = 0.50
    inactive_reid_threshold: float = 0.62
    min_inactive_color_similarity: float = 0.22
    min_inactive_shape_similarity: float = 0.48
    inactive_confident_feature_similarity: float = 0.70
    inactive_confident_color_similarity: float = 0.30
    inactive_confident_shape_similarity: float = 0.62
    inactive_match_min_score: float = 2.15
    inactive_match_row_margin: float = 0.55
    inactive_match_col_margin: float = 0.55
    inactive_long_gap_frames: int = 90
    inactive_long_gap_extra_feature: float = 0.07
    inactive_min_reuse_hits: int = 4
    archive_min_hits: int = 5
    archive_min_reuse_hits: int = 6
    max_archive_age: int = 3600
    archive_reid_threshold: float = 0.64
    archive_color_threshold: float = 0.24
    archive_shape_threshold: float = 0.58
    archive_match_min_score: float = 2.20
    archive_match_row_margin: float = 0.75
    archive_match_col_margin: float = 0.75
    feature_bank_size: int = 12
    color_bank_size: int = 12
    shape_bank_size: int = 12
    identity_feature_momentum: float = 0.96
    identity_color_momentum: float = 0.94
    identity_shape_momentum: float = 0.95
    identity_update_min_similarity: float = 0.26
    identity_color_update_min_similarity: float = 0.10
    identity_bank_topk: int = 3

    reid_backbone: str = "resnet50_gem"
    reid_weights: str = "reid_resnet50_msmt17.pth"
    reid_input_height: int = 256
    reid_input_width: int = 128
    reid_padding: float = 0.08
    reid_interval: int = 2
    reid_force_count: int = 2
    max_reid_detections: int = 4
    reid_num_parts: int = 3
    reid_gem_p: float = 3.0
    reid_global_weight: float = 1.0
    reid_part_weight: float = 0.65
    reid_hist_h_bins: int = 12
    reid_hist_s_bins: int = 8

    prediction_center_alpha: float = 0.16
    prediction_size_alpha: float = 0.08
    measurement_center_alpha: float = 0.42
    measurement_size_alpha: float = 0.18
    size_jump_ratio: float = 0.18
    center_deadzone_px: int = 6
    size_deadzone_px: int = 6

    detector_duplicate_iou: float = 0.58
    detector_nested_overlap: float = 0.76
    track_duplicate_iou: float = 0.64
    track_nested_overlap: float = 0.82

    route_log_min_distance: int = 4

    writer_default_fps: float = 25.0
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "tracking_output"
    )
