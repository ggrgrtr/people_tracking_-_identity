import json
import math
import time

import cv2
import numpy as np

from .utils import get_center, track_color


class EventLogger:
    def __init__(self, config, source_label):
        self.config = config
        self.source_label = source_label
        self.people = {}
        self.last_frame_shape = None

    def _get_person_entry(self, track, frame_id, elapsed_seconds):
        if track.id not in self.people:
            self.people[track.id] = {
                "track_id": track.id,
                "first_seen_frame": frame_id,
                "first_seen_time_sec": round(elapsed_seconds, 2),
                "last_seen_frame": frame_id,
                "last_seen_time_sec": round(elapsed_seconds, 2),
                "frames_visible": 0,
                "distance_px": 0.0,
                "trajectory": [],
            }

        return self.people[track.id]

    def _record_route(self, track, frame_id, elapsed_seconds):
        person = self._get_person_entry(track, frame_id, elapsed_seconds)
        center = get_center(track.smooth_bbox)
        bbox = [int(value) for value in track.smooth_bbox]

        person["last_seen_frame"] = frame_id
        person["last_seen_time_sec"] = round(elapsed_seconds, 2)
        person["frames_visible"] += 1

        should_append = True
        if person["trajectory"]:
            prev_center = person["trajectory"][-1]["center"]
            distance = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
            if distance < self.config.route_log_min_distance:
                should_append = False
            else:
                person["distance_px"] += distance

        if should_append:
            person["trajectory"].append(
                {
                    "frame": frame_id,
                    "time_sec": round(elapsed_seconds, 2),
                    "center": [int(center[0]), int(center[1])],
                    "bbox": bbox,
                }
            )

    def process_tracks(self, tracks, frame_id, elapsed_seconds, frame_shape):
        self.last_frame_shape = frame_shape

        for track in tracks:
            if not track.is_confirmed(self.config.min_confirmed_hits):
                continue

            self._record_route(track, frame_id, elapsed_seconds)

    def _route_image_name(self, track_id):
        return f"track_{int(track_id):03d}_route.png"

    def _draw_route_image(self, trajectory, frame_shape, track_id):
        if frame_shape is None:
            frame_shape = (self.config.camera_height, self.config.camera_width, 3)

        height, width = frame_shape[:2]
        canvas = np.full((height, width, 3), 28, dtype=np.uint8)

        if trajectory:
            points = [tuple(point["center"]) for point in trajectory]
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
                cv2.arrowedLine(
                    canvas,
                    points[-2],
                    points[-1],
                    (245, 245, 245),
                    3,
                    tipLength=0.25,
                )

        cv2.putText(
            canvas,
            f"track_id: {track_id}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            canvas,
            "Route: darker -> lighter over time",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 128, 255),
            1,
        )
        cv2.putText(
            canvas,
            "Green -> Red",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 128, 255),
            1,
        )

        return canvas

    def save_route_images(self, routes_dir):
        route_paths = {}

        for track_id in sorted(self.people):
            person = self.people[track_id]
            filename = self._route_image_name(track_id)
            image_path = routes_dir / filename
            route_image = self._draw_route_image(
                person["trajectory"],
                self.last_frame_shape,
                track_id,
            )
            cv2.imwrite(str(image_path), route_image)
            route_paths[track_id] = str(image_path)

        return route_paths

    def save(self, output_path, routes_dir=None, session_duration=None):
        route_paths = self.save_route_images(routes_dir) if routes_dir is not None else {}
        people = []
        for track_id in sorted(self.people):
            person = self.people[track_id]
            people.append(
                {
                    "track_id": person["track_id"],
                    "first_seen_frame": person["first_seen_frame"],
                    "first_seen_time_sec": person["first_seen_time_sec"],
                    "last_seen_frame": person["last_seen_frame"],
                    "last_seen_time_sec": person["last_seen_time_sec"],
                    "frames_visible": person["frames_visible"],
                    "distance_px": round(person["distance_px"], 2),
                    "trajectory_points": len(person["trajectory"]),
                    "route_image": route_paths.get(track_id, ""),
                }
            )

        payload = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": self.source_label,
            "people": people,
        }
        if session_duration is not None:
            payload["session_duration_sec"] = round(session_duration, 2)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
