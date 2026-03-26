# создание скриптов
# для работы с аргументами командной строки, парсинг аргументов
import argparse

# для построения и управления путями
from pathlib import Path

# многопоточный ввод
import threading
import time

import cv2
import torch

# из первого экземпляра программы берем детектор, определение признаков и сборщик
from people_tracking.detector import PersonDetector
from people_tracking.reid import AppearanceEncoder
from people_tracking.utils import RateMeter, build_output_paths, is_plausible_fps, resolve_source

from .config import AppConfig
from .face_backend import OptionalFaceBackend
from .identity_manager import IdentityManager
from .renderer import draw_dashboard, draw_tracklet
from .tracklets import TrackletTracker


# для 1 объекта мониторинга камеры
class ThreadedCameraCapture:
    def __init__(self, capture):
        self.capture = capture
        self.condition = threading.Condition()
        self.stopped = False
        self.read_failed = False
        self.max_consecutive_failures = 3
        self.latest_frame = None
        self.latest_timestamp = None
        self.latest_seq = 0
        self.last_consumed_seq = 0
        self.thread = None

    def start(self):
        if self.thread is not None:
            return self
        if not hasattr(threading, "_start_new_thread"):
            return None
        self.thread = threading.Thread(target=self._reader, daemon=True)
        try:
            self.thread.start()
        except Exception:
            self.thread = None
            return None
        return self

    def _reader(self):
        consecutive_failures = 0
        while not self.stopped:
            ok, frame = self.capture.read()
            timestamp = time.perf_counter()
            with self.condition:
                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures >= self.max_consecutive_failures:
                        self.read_failed = True
                        self.stopped = True
                    self.condition.notify_all()
                    if self.read_failed:
                        return
                else:
                    consecutive_failures = 0
                    self.latest_frame = frame
                    self.latest_timestamp = timestamp
                    self.latest_seq += 1
                    self.condition.notify_all()
            if not ok:
                time.sleep(0.05)

    def read(self, timeout=1.0):
        deadline = time.perf_counter() + max(0.01, float(timeout))
        with self.condition:
            while (
                not self.read_failed
                and not self.stopped
                and self.latest_seq == self.last_consumed_seq
            ):
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                self.condition.wait(timeout=min(0.05, remaining))

            if self.latest_seq != self.last_consumed_seq:
                self.last_consumed_seq = self.latest_seq
                return "frame", self.latest_frame, self.latest_timestamp
            if self.read_failed or self.stopped:
                return "eof", None, None
            return "timeout", None, None

    def stop(self):
        self.stopped = True
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identity gallery tracking: tracklets + persistent person identities"
    )
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
    threaded_capture = None
    if is_live_source and config.threaded_camera_capture:
        # Для live-источника пытаемся отделить чтение камеры от тяжелой обработки кадров.
        threaded_capture = ThreadedCameraCapture(cap).start()
        if threaded_capture is None:
            # Если среда не умеет поднимать потоки, программа должна деградировать мягко, а не падать.
            print("Warning: threaded camera capture is unavailable, using synchronous capture.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    base_dir = Path(__file__).resolve().parents[1]
    detector = PersonDetector(config, base_dir, device)
    encoder = AppearanceEncoder(config, device, base_dir)
    face_backend = OptionalFaceBackend(config, base_dir)
    tracker = TrackletTracker(config)
    identity_manager = IdentityManager(config)

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
    print(f"Face backend: {face_backend.description}")
    if should_save_output:
        print(f"Output video: {output_video_path}")
    print(f"Identity log: {output_paths['events']}")
    print("Q/q - quit")

    while True:
        if threaded_capture is not None:
            read_status, frame, read_complete_time = threaded_capture.read(timeout=1.0)
            if read_status == "timeout":
                continue
            if read_status != "frame":
                break
            if read_complete_time is None:
                read_complete_time = time.perf_counter()
        else:
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
            source_fps = read_cadence_fps if read_cadence_fps > 0.0 else metadata_source_fps
        elif metadata_source_fps > 0.0:
            source_fps = metadata_source_fps
        elif pts_based_fps > 0.0:
            source_fps = pts_based_fps
        else:
            source_fps = read_cadence_fps

        frame_id += 1
        elapsed_seconds = time.time() - session_start

        has_active_tracklets = bool(tracker.active_tracklets)
        # YOLO запускается разреженно: локальное движение между detect-pass ведет Kalman tracker.
        detect_interval = config.yolo_interval if has_active_tracklets else config.empty_scene_yolo_interval
        should_detect = frame_id == 1 or frame_id % max(1, detect_interval) == 0

        finished_tracklets = []
        if should_detect:
            detect_pass_count += 1
            detections = detector.detect(frame)
            # Дорогой ReID считаем не для всех рамок подряд, а только для тех, где он реально нужен.
            candidate_feature_indices = tracker.reid_candidate_detection_indices(
                detections,
                frame.shape,
            )
            has_pending_identities = any(
                track.is_confirmed() and track.person_id is None
                for track in tracker.active_tracklets
            )
            need_full_reid = (
                not has_active_tracklets
                or has_pending_identities
                or len(candidate_feature_indices) >= config.reid_force_count
            )
            should_extract_features = bool(detections) and (
                need_full_reid
                or detect_pass_count % max(1, config.reid_interval) == 0
            )
            selected_feature_indices = None if should_extract_features else []
            if should_extract_features and not need_full_reid:
                selected_feature_indices = candidate_feature_indices[: config.max_reid_detections]
                if not selected_feature_indices:
                    selected_feature_indices = sorted(
                        range(len(detections)),
                        key=lambda index: max(1, int(detections[index][2])) * max(1, int(detections[index][3])),
                        reverse=True,
                    )[: config.max_reid_detections]

            features, color_histograms = encoder.extract(
                frame,
                detections,
                include_features=should_extract_features,
                max_feature_boxes=None,
                feature_indices=selected_feature_indices,
            )
            face_features = face_backend.extract_batch(
                frame,
                detections,
                candidate_indices=selected_feature_indices,
            )
            finished_tracklets = tracker.update(
                detections,
                features,
                color_histograms,
                face_features,
                frame.shape,
            )
        else:
            finished_tracklets = tracker.predict_only(frame.shape)

        identity_manager.finalize_tracklets(finished_tracklets)
        visible_tracklets = tracker.visible_tracklets()
        # В identity-логику передаем только реально наблюденные tracklet, а не prediction-only кадры.
        observed_tracklets = [track for track in visible_tracklets if track.was_observed()]
        identity_manager.observe_tracklets(
            observed_tracklets,
            frame_id,
            elapsed_seconds,
            frame.shape,
            update_gallery=should_detect,
        )

        display_frame = frame.copy()
        for tracklet in visible_tracklets:
            draw_tracklet(display_frame, tracklet)

        if not visible_tracklets:
            cv2.putText(
                display_frame,
                "No people detected",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        identified_tracks = sum(1 for tracklet in visible_tracklets if tracklet.person_id is not None)
        draw_dashboard(
            display_frame,
            source_fps,
            len(visible_tracklets),
            identified_tracks,
            should_save_output,
        )

        if writer is None and should_save_output:
            writer_source_fps = cap.get(cv2.CAP_PROP_FPS)
            writer_fps = writer_source_fps if writer_source_fps and writer_source_fps > 1 else config.writer_default_fps
            writer = create_writer(output_video_path, writer_fps, display_frame.shape)

        if writer is not None:
            writer.write(display_frame)

        if not args.no_display:
            cv2.imshow("Identity Gallery Tracking", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

    remaining_tracklets = tracker.finish_all()
    identity_manager.finalize_tracklets(remaining_tracklets)

    if threaded_capture is not None:
        threaded_capture.stop()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    session_duration = time.time() - session_start
    identity_manager.save(
        output_paths["events"],
        routes_dir=output_paths["routes_dir"],
        session_duration=session_duration,
    )
    print(f"Saved identity log to: {output_paths['events']}")
    print(f"Saved session outputs to: {output_paths['session_dir']}")
    if writer is not None:
        print(f"Saved tracked video to: {output_video_path}")


if __name__ == "__main__":
    main()
