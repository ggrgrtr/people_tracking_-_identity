from collections import deque
from pathlib import Path
import math
import time


class RateMeter:
    def __init__(self, alpha=0.12):
        self.alpha = float(alpha)
        self.value = 0.0
        self.prev_timestamp = None

    def reset(self):
        self.value = 0.0
        self.prev_timestamp = None

    def update(self, timestamp=None):
        if timestamp is None:
            timestamp = time.perf_counter()

        timestamp = float(timestamp)
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
            return self.value

        dt = timestamp - self.prev_timestamp
        self.prev_timestamp = timestamp
        return self.update_delta(dt)

    def update_delta(self, dt):
        dt = float(dt)
        if dt <= 0.0:
            return self.value

        instant_rate = 1.0 / dt
        if self.value <= 0.0:
            self.value = instant_rate
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * instant_rate
        return self.value


def sanitize_name(text):
    cleaned = []
    for char in text:
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        else:
            cleaned.append("_")

    value = "".join(cleaned).strip("_")
    return value or "source"


def build_output_paths(output_dir, source_label):
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    source_name = sanitize_name(source_label)
    session_dir = output_dir / f"{source_name}_{stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    return {
        "session_dir": session_dir,
        "video": session_dir / f"{source_name}_tracked.mp4",
        "events": session_dir / f"{source_name}_events.json",
        "routes_dir": session_dir,
    }


def resolve_source(source_text):
    if source_text.isdigit():
        return int(source_text), f"camera_{source_text}"

    source_path = Path(source_text)
    return str(source_path), source_path.stem or "video"


def resolve_yolo_weights(base_dir):
    candidates = [
        base_dir / "yolov8n.pt",
        base_dir.parent / "yolov8n.pt",
        Path.cwd() / "yolov8n.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return "yolov8n.pt"


def is_plausible_fps(value, min_fps=1.0, max_fps=120.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False

    return min_fps <= value <= max_fps


def resolve_reid_weights(base_dir, explicit_path=""):
    candidates = []

    if explicit_path:
        explicit = Path(explicit_path)
        candidates.append(explicit)
        if not explicit.is_absolute():
            candidates.append(base_dir / explicit)
            candidates.append(Path.cwd() / explicit)

    candidates.extend(
        [
            base_dir / "reid_resnet50_market1501.pth",
            base_dir / "reid_resnet50_msmt17.pth",
            base_dir / "reid_resnet50_dukemtmc.pth",
            base_dir / "person_reid_resnet50.pth",
            base_dir / "reid_resnet50.pth",
            base_dir / "weights" / "reid_resnet50_market1501.pth",
            base_dir / "weights" / "reid_resnet50_msmt17.pth",
            base_dir.parent / "reid_resnet50_market1501.pth",
            base_dir.parent / "reid_resnet50_msmt17.pth",
            Path.cwd() / "reid_resnet50_market1501.pth",
            Path.cwd() / "reid_resnet50_msmt17.pth",
        ]
    )

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.is_absolute() else candidate
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    return None


def clip_bbox(bbox, frame_shape):
    x, y, w, h = [int(v) for v in bbox]
    frame_h, frame_w = frame_shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)

    return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def box_valid(bbox, frame_shape, min_box_w, min_box_h):
    if bbox is None:
        return False

    _, _, w, h = clip_bbox(bbox, frame_shape)
    return w >= min_box_w and h >= min_box_h


def get_center(bbox):
    x, y, w, h = [int(v) for v in bbox]
    return (x + w // 2, y + h // 2)


def point_in_bbox(point, bbox):
    px, py = [int(v) for v in point]
    x, y, w, h = [int(v) for v in bbox]
    return x <= px <= x + w and y <= py <= y + h


def bbox_from_center(center, w, h):
    cx, cy = center
    return (int(cx - w // 2), int(cy - h // 2), int(w), int(h))


def compute_iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0

    x1, y1, w1, h1 = [int(v) for v in box1]
    x2, y2, w2, h2 = [int(v) for v in box2]

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    union = w1 * h1 + w2 * h2 - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def bbox_area(bbox):
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def intersection_over_smaller(box1, box2):
    if box1 is None or box2 is None:
        return 0.0

    x1, y1, w1, h1 = [int(v) for v in box1]
    x2, y2, w2, h2 = [int(v) for v in box2]

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h
    smaller_area = min(max(1, w1 * h1), max(1, w2 * h2))

    return inter_area / smaller_area


def center_distance(box1, box2):
    cx1, cy1 = get_center(box1)
    cx2, cy2 = get_center(box2)
    return math.hypot(cx1 - cx2, cy1 - cy2)


def size_similarity(box1, box2):
    w1, h1 = int(box1[2]), int(box1[3])
    w2, h2 = int(box2[2]), int(box2[3])

    if min(w1, h1, w2, h2) <= 0:
        return 0.0

    width_ratio = min(w1, w2) / max(w1, w2)
    height_ratio = min(h1, h2) / max(h1, h2)
    return (width_ratio + height_ratio) * 0.5


def smooth_point(prev_point, new_point, alpha):
    if prev_point is None:
        return (int(new_point[0]), int(new_point[1]))

    px, py = prev_point
    nx, ny = new_point
    return (int(round(px + alpha * (nx - px))), int(round(py + alpha * (ny - py))))


def smooth_size(prev_size, new_size, alpha, min_box_w, min_box_h):
    if prev_size is None:
        return (max(min_box_w, int(new_size[0])), max(min_box_h, int(new_size[1])))

    pw, ph = prev_size
    nw, nh = new_size
    sw = int(round(pw + alpha * (nw - pw)))
    sh = int(round(ph + alpha * (nh - ph)))
    return (max(min_box_w, sw), max(min_box_h, sh))


def stabilize_size(prev_size, new_size, size_jump_ratio, min_box_w, min_box_h):
    if prev_size is None:
        return (max(min_box_w, int(new_size[0])), max(min_box_h, int(new_size[1])))

    pw, ph = prev_size
    nw, nh = new_size

    if pw <= 0 or ph <= 0:
        return (max(min_box_w, int(new_size[0])), max(min_box_h, int(new_size[1])))

    min_w = max(min_box_w, int(pw * (1.0 - size_jump_ratio)))
    max_w = int(pw * (1.0 + size_jump_ratio))
    min_h = max(min_box_h, int(ph * (1.0 - size_jump_ratio)))
    max_h = int(ph * (1.0 + size_jump_ratio))

    stable_w = min(max(int(nw), min_w), max_w)
    stable_h = min(max(int(nh), min_h), max_h)
    return (stable_w, stable_h)


def smooth_bbox(
    prev_bbox,
    new_bbox,
    frame_shape,
    center_alpha,
    size_alpha,
    size_jump_ratio,
    min_box_w,
    min_box_h,
    center_deadzone_px=0,
    size_deadzone_px=0,
):
    new_bbox = clip_bbox(new_bbox, frame_shape)
    if prev_bbox is None:
        return new_bbox

    prev_center = get_center(prev_bbox)
    new_center = get_center(new_bbox)
    center_shift = math.hypot(new_center[0] - prev_center[0], new_center[1] - prev_center[1])
    effective_center_alpha = center_alpha
    if center_shift <= max(1, center_deadzone_px):
        effective_center_alpha *= 0.65

    if center_shift <= 1.0:
        smooth_center = prev_center
    else:
        smooth_center = smooth_point(prev_center, new_center, effective_center_alpha)

    prev_size = (int(prev_bbox[2]), int(prev_bbox[3]))
    new_size = (int(new_bbox[2]), int(new_bbox[3]))
    stable_size = stabilize_size(prev_size, new_size, size_jump_ratio, min_box_w, min_box_h)
    size_shift = max(
        abs(stable_size[0] - prev_size[0]),
        abs(stable_size[1] - prev_size[1]),
    )
    effective_size_alpha = size_alpha
    if size_shift <= max(1, size_deadzone_px):
        effective_size_alpha *= 0.65

    if size_shift <= 1:
        smooth_wh = (
            max(min_box_w, prev_size[0]),
            max(min_box_h, prev_size[1]),
        )
    else:
        smooth_wh = smooth_size(
            prev_size,
            stable_size,
            effective_size_alpha,
            min_box_w,
            min_box_h,
        )

    return clip_bbox(
        bbox_from_center(smooth_center, smooth_wh[0], smooth_wh[1]),
        frame_shape,
    )


def append_path(path, point, min_distance):
    if not path:
        path.append(point)
        return

    prev_x, prev_y = path[-1]
    if math.hypot(point[0] - prev_x, point[1] - prev_y) >= min_distance:
        path.append(point)


def nearest_frame_edge(bbox, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = [int(v) for v in bbox]

    distances = {
        "left": x,
        "right": frame_w - (x + w),
        "top": y,
        "bottom": frame_h - (y + h),
    }

    edge_name, edge_distance = min(distances.items(), key=lambda item: item[1])
    edge_margin = max(24, int(min(frame_w, frame_h) * 0.08))
    return edge_name if edge_distance <= edge_margin else None


def track_color(track_id):
    return (
        int((37 * track_id) % 205) + 50,
        int((79 * track_id) % 205) + 50,
        int((131 * track_id) % 205) + 50,
    )
