import cv2

from .utils import track_color


def draw_track(frame, track):
    x, y, w, h = [int(v) for v in track.smooth_bbox]
    color = track_color(track.id)
    label = f"ID {track.id}"
    if not track.is_confirmed(track.min_confirmed_hits):
        label += " ..."

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(24, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )

    points = list(track.path)
    for index in range(1, len(points)):
        cv2.line(frame, points[index - 1], points[index], color, 2)

    if len(points) >= 2:
        cv2.arrowedLine(frame, points[-2], points[-1], color, 2, tipLength=0.25)


def draw_dashboard(frame, source_fps, tracks_count, recording):
    cv2.putText(
        frame,
        f"CAM: {source_fps:.1f}",
        (frame.shape[1] - 190, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Tracks: {tracks_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if recording:
        cv2.putText(
            frame,
            "REC",
            (frame.shape[1] - 90, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
