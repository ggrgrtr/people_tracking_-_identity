import cv2

from people_tracking.utils import track_color


def draw_tracklet(frame, tracklet):
    x, y, w, h = [int(v) for v in tracklet.smooth_bbox]
    color_key = tracklet.person_id if tracklet.person_id is not None else tracklet.id
    color = track_color(color_key)

    label = f"ID {tracklet.person_id}" if tracklet.person_id is not None else f"Track {tracklet.id}"
    if tracklet.person_id is None and tracklet.is_confirmed(tracklet.min_confirmed_hits):
        label += " pending"
    elif not tracklet.is_confirmed(tracklet.min_confirmed_hits):
        label += " ..."

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(24, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        color,
        2,
    )

    points = list(tracklet.path)
    for index in range(1, len(points)):
        cv2.line(frame, points[index - 1], points[index], color, 2)
    if len(points) >= 2:
        cv2.arrowedLine(frame, points[-2], points[-1], color, 2, tipLength=0.25)


def draw_dashboard(frame, source_fps, active_tracks, identified_tracks, recording):
    cv2.putText(
        frame,
        f"FPS: {source_fps:.1f}",
        (frame.shape[1] - 190, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Tracklets: {active_tracks}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Identified: {identified_tracks}",
        (20, 60),
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
