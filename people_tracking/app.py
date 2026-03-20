import argparse
from pathlib import Path
import time

import cv2
import torch

from .config import AppConfig
from .detector import PersonDetector
from .events import EventLogger
from .reid import AppearanceEncoder
from .renderer import draw_dashboard, draw_track
from .tracker import MultiObjectTracker
from .utils import RateMeter, build_output_paths, is_plausible_fps, resolve_source


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-person tracking with Kalman + Hungarian")
    parser.add_argument("--source", default="0", help="Camera index like 0 or path to a video file")
    parser.add_argument("--save-output", action="store_true", help="Save the rendered result to video")
    parser.add_argument("--output", default="", help="Optional explicit output video path")
    parser.add_argument("--no-display", action="store_true", help="Run without preview window")
    return parser.parse_args()


def open_capture(source, config):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    return cv2.VideoCapture(source)


def create_writer(path, fps, frame_shape):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (frame_shape[1], frame_shape[0]))
    return writer if writer.isOpened() else None


def main():
    args = parse_args()
    config = AppConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    source, source_label = resolve_source(args.source)
    is_live_source = isinstance(source, int)
    cap = open_capture(source, config)
    if not cap.isOpened():
        print("Failed to open source.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    base_dir = Path(__file__).resolve().parents[1]
    detector = PersonDetector(config, base_dir, device)
    encoder = AppearanceEncoder(config, device, base_dir)
    tracker = MultiObjectTracker(config)
    events = EventLogger(config, source_label)

    output_paths = build_output_paths(config.output_dir, source_label)
    should_save_output = args.save_output or not is_live_source or bool(args.output)
    output_video_path = Path(args.output) if args.output else output_paths["video"]
    if args.output:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    frame_id = 0
    detect_pass_count = 0
    session_start = time.time()
    source_read_meter = RateMeter(alpha=0.12)
    source_pts_meter = RateMeter(alpha=0.12)
    source_fps = 0.0
    last_pos_msec = None
    declared_source_fps = cap.get(cv2.CAP_PROP_FPS)
    metadata_source_fps = declared_source_fps if is_plausible_fps(declared_source_fps) else 0.0

    print(f"Device: {device}")
    print(f"Source: {source_label}")
    print(f"Session folder: {output_paths['session_dir']}")
    print(f"ReID: {encoder.description}")
    if should_save_output:
        print(f"Output video: {output_video_path}")
    print(f"Event log: {output_paths['events']}")
    print("Q/q - quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        read_complete_time = time.perf_counter()

        read_cadence_fps = source_read_meter.update(read_complete_time)
        pts_based_fps = source_pts_meter.value

        if is_live_source and config.mirror_camera:
            frame = cv2.flip(frame, 1)
        elif metadata_source_fps <= 0.0:
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec is not None and pos_msec > 0.0:
                if last_pos_msec is not None:
                    pts_based_fps = source_pts_meter.update_delta((pos_msec - last_pos_msec) / 1000.0)
                last_pos_msec = pos_msec

        if is_live_source:
            if read_cadence_fps > 0.0:
                source_fps = read_cadence_fps
            else:
                source_fps = metadata_source_fps
        elif metadata_source_fps > 0.0:
            source_fps = metadata_source_fps
        elif pts_based_fps > 0.0:
            source_fps = pts_based_fps
        else:
            source_fps = read_cadence_fps

        frame_id += 1
        elapsed_seconds = time.time() - session_start

        has_active_tracks = bool(tracker.active_tracks)
        detect_interval = config.yolo_interval if has_active_tracks else config.empty_scene_yolo_interval
        should_detect = frame_id == 1 or frame_id % max(1, detect_interval) == 0
        if should_detect:
            detect_pass_count += 1
            detections = detector.detect(frame)
            should_extract_features = (
                bool(detections)
                and (
                    not has_active_tracks
                    or bool(tracker.inactive_tracks)
                    or bool(tracker.archived_tracks)
                    or len(detections) >= config.reid_force_count
                    or detect_pass_count % max(1, config.reid_interval) == 0
                )
            )
            feature_limit = (
                None
                if (
                    not should_extract_features
                    or bool(tracker.inactive_tracks)
                    or bool(tracker.archived_tracks)
                    or len(detections) >= config.reid_force_count
                )
                else config.max_reid_detections
            )
            features, color_histograms = encoder.extract(
                frame,
                detections,
                include_features=should_extract_features,
                max_feature_boxes=feature_limit,
            )
            tracker.update(detections, features, color_histograms, frame.shape)
        else:
            tracker.predict_only(frame.shape)

        visible_tracks = tracker.visible_tracks()
        events.process_tracks(visible_tracks, frame_id, elapsed_seconds, frame.shape)

        display_frame = frame.copy()

        for track in visible_tracks:
            draw_track(display_frame, track)

        if not visible_tracks:
            cv2.putText(
                display_frame,
                "No people detected",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        draw_dashboard(
            display_frame,
            source_fps,
            len(visible_tracks),
            should_save_output,
        )

        if writer is None and should_save_output:
            writer_source_fps = cap.get(cv2.CAP_PROP_FPS)
            writer_fps = (
                writer_source_fps
                if writer_source_fps and writer_source_fps > 1
                else config.writer_default_fps
            )
            writer = create_writer(output_video_path, writer_fps, display_frame.shape)

        if writer is not None:
            writer.write(display_frame)

        if not args.no_display:
            cv2.imshow("Kalman Hungarian People Tracking", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    session_duration = time.time() - session_start
    events.save(
        output_paths["events"],
        routes_dir=output_paths["routes_dir"],
        session_duration=session_duration,
    )
    print(f"Saved event log to: {output_paths['events']}")
    print(f"Saved session outputs to: {output_paths['session_dir']}")
    if writer is not None:
        print(f"Saved tracked video to: {output_video_path}")
