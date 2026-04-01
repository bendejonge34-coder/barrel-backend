import os
import sys
import json
import cv2
import math
import contextlib
import io
import traceback

# Must be set BEFORE importing ultralytics
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["PYTHONUNBUFFERED"] = "1"

from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BARREL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "runs_detect",
    "barrel_detector_local",
    "weights",
    "best.pt",
)

HORSE_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

HORSE_CLASS_ID = 17  # COCO horse
BARREL_LABELS = ["barrel1", "barrel2", "barrel3"]

# Lean MVP performance settings for stability
HORSE_CONFIDENCE_THRESHOLD = 0.18
BARREL_CONFIDENCE_THRESHOLD = 0.50

TARGET_SAMPLE_FPS = 3.0
MAX_SAMPLED_FRAMES = 42
MIN_SAMPLED_FRAMES = 18

MAX_INFERENCE_WIDTH = 640
MAX_INFERENCE_HEIGHT = 360

SMOOTHING_ALPHA = 0.42
INTERPOLATION_MAX_GAP = 2

PATTERN_APEX_TIME_TOLERANCE_SECONDS = 0.22

# Canonical pattern coordinates
CANONICAL_LEFT_BARREL_X = 0.20
CANONICAL_RIGHT_BARREL_X = 0.80
CANONICAL_TOP_BARREL_X = 0.50

CANONICAL_TOP_BARREL_Y = 0.18
CANONICAL_LOWER_BARREL_Y = 0.72
CANONICAL_HOME_Y = 1.02


def log_err(*args):
    print(*args, file=sys.stderr, flush=True)


def emit_json(payload):
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def fail(error_message, extra=None):
    payload = {"ok": False, "error": error_message}
    if extra and isinstance(extra, dict):
        payload.update(extra)
    emit_json(payload)


def distance(p1, p2):
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def round_point(point, decimals=2):
    if point is None:
        return None
    return [round(float(point[0]), decimals), round(float(point[1]), decimals)]


def round_or_none(value, decimals=2):
    if value is None:
        return None
    return round(float(value), decimals)


def compute_bottom_center(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = y2
    return cx, cy


def safe_imwrite(output_path, image):
    try:
        if image is None:
            return False
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return bool(cv2.imwrite(output_path, image))
    except Exception:
        return False


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return YOLO(model_path)


def resize_for_inference(frame):
    if frame is None:
        return None, 1.0, 1.0

    original_height, original_width = frame.shape[:2]

    if original_width <= 0 or original_height <= 0:
        return frame, 1.0, 1.0

    width_scale = MAX_INFERENCE_WIDTH / float(original_width)
    height_scale = MAX_INFERENCE_HEIGHT / float(original_height)
    scale = min(width_scale, height_scale, 1.0)

    if scale >= 1.0:
        return frame, 1.0, 1.0

    new_width = max(1, int(round(original_width * scale)))
    new_height = max(1, int(round(original_height * scale)))

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    x_ratio = original_width / float(new_width)
    y_ratio = original_height / float(new_height)

    return resized, x_ratio, y_ratio


def detect_barrels_in_frame(frame, barrel_model, confidence_threshold=BARREL_CONFIDENCE_THRESHOLD):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        results = barrel_model.predict(
            source=frame,
            conf=confidence_threshold,
            verbose=False,
        )

    barrels = []

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return barrels

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0].item())

        barrels.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": round(conf, 3),
            "center_x": float((x1 + x2) / 2.0),
            "center_y": float((y1 + y2) / 2.0),
        })

    return barrels


def scale_barrel_detections_to_original(barrels, x_ratio, y_ratio):
    scaled = []

    for barrel in barrels:
        scaled.append({
            "x1": int(round(float(barrel["x1"]) * x_ratio)),
            "y1": int(round(float(barrel["y1"]) * y_ratio)),
            "x2": int(round(float(barrel["x2"]) * x_ratio)),
            "y2": int(round(float(barrel["y2"]) * y_ratio)),
            "confidence": barrel["confidence"],
            "center_x": int(round(float(barrel["center_x"]) * x_ratio)),
            "center_y": int(round(float(barrel["center_y"]) * y_ratio)),
        })

    return scaled


def build_horse_candidates(result, x_ratio=1.0, y_ratio=1.0):
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes_xyxy = result.boxes.xyxy.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()

    candidates = []

    for box, cls_id, conf in zip(boxes_xyxy, classes, confidences):
        if int(cls_id) != HORSE_CLASS_ID:
            continue

        x1, y1, x2, y2 = box
        x1 *= x_ratio
        y1 *= y_ratio
        x2 *= x_ratio
        y2 *= y_ratio

        cx, cy = compute_bottom_center((x1, y1, x2, y2))
        area = max(0.0, (x2 - x1) * (y2 - y1))
        height = max(0.0, y2 - y1)

        # Slightly above absolute bottom is usually a more stable rider/horse path point
        tracking_point = (float(cx), float(y2 - height * 0.12))

        candidates.append({
            "confidence": round(float(conf), 4),
            "bbox": [
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2), 2),
                round(float(y2), 2),
            ],
            "center": [round(float(cx), 2), round(float(cy), 2)],
            "tracking_point": tracking_point,
            "area": float(area),
        })

    return candidates


def choose_best_candidate(candidates, prev_point, frame_width, frame_height):
    if not candidates:
        return None

    if prev_point is None:
        return max(candidates, key=lambda c: (c["confidence"], c["area"]))

    diagonal = math.hypot(frame_width, frame_height) if frame_width > 0 and frame_height > 0 else 1.0

    best_candidate = None
    best_score = None

    for candidate in candidates:
        dist = distance(prev_point, candidate["tracking_point"])
        normalized_dist = dist / diagonal
        score = (candidate["confidence"] * 1.8) + min(candidate["area"] / 50000.0, 1.0) - (normalized_dist * 2.4)

        if best_score is None or score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def smooth_points(points, alpha=SMOOTHING_ALPHA):
    if not points:
        return []

    smoothed = [points[0]]

    for i in range(1, len(points)):
        prev = smoothed[-1]
        cur = points[i]
        sx = alpha * float(cur[0]) + (1.0 - alpha) * float(prev[0])
        sy = alpha * float(cur[1]) + (1.0 - alpha) * float(prev[1])
        smoothed.append((sx, sy))

    return smoothed


def interpolate_small_gaps(track_points, max_gap=INTERPOLATION_MAX_GAP):
    if not track_points:
        return []

    result = list(track_points)
    n = len(result)
    i = 0

    while i < n:
        if result[i] is not None:
            i += 1
            continue

        gap_start = i - 1
        j = i
        while j < n and result[j] is None:
            j += 1
        gap_end = j

        gap_len = gap_end - i

        if (
            gap_start >= 0
            and gap_end < n
            and result[gap_start] is not None
            and result[gap_end] is not None
            and gap_len <= max_gap
        ):
            p1 = result[gap_start]
            p2 = result[gap_end]

            for k in range(1, gap_len + 1):
                t = k / float(gap_len + 1)
                x = float(p1[0]) + t * (float(p2[0]) - float(p1[0]))
                y = float(p1[1]) + t * (float(p2[1]) - float(p1[1]))
                result[gap_start + k] = (x, y)

        i = gap_end

    return result


def adaptive_max_jump(width, height):
    diagonal = math.hypot(width, height) if width > 0 and height > 0 else 1000.0
    return clamp(diagonal * 0.13, 110.0, 320.0)


def dedupe_points(points, min_dist=1e-6):
    if not points:
        return []

    deduped = [points[0]]
    for pt in points[1:]:
        if distance(pt, deduped[-1]) > min_dist:
            deduped.append(pt)
    return deduped


def normalize_points_to_unit_box(points, padding=0.08):
    if not points:
        return []

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    normalized = []

    for x, y in points:
        nx = (float(x) - min_x) / range_x
        ny = (float(y) - min_y) / range_y

        nx = padding + nx * (1.0 - 2.0 * padding)
        ny = padding + ny * (1.0 - 2.0 * padding)

        normalized.append([round(nx, 4), round(ny, 4)])

    return normalized


def build_sample_frame_indices(frame_count, fps):
    if frame_count <= 0:
        return []

    if fps <= 0:
        fps = 30.0

    step = max(1, int(round(fps / TARGET_SAMPLE_FPS)))
    indices = list(range(0, frame_count, step))

    if not indices:
        indices = [0]

    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)

    if len(indices) > MAX_SAMPLED_FRAMES:
        desired = MAX_SAMPLED_FRAMES
        indices = sorted(
            set(int(round(i * (frame_count - 1) / max(desired - 1, 1))) for i in range(desired))
        )

    if len(indices) < MIN_SAMPLED_FRAMES and frame_count > MIN_SAMPLED_FRAMES:
        desired = min(MIN_SAMPLED_FRAMES, frame_count)
        indices = sorted(
            set(int(round(i * (frame_count - 1) / max(desired - 1, 1))) for i in range(desired))
        )

    return sorted(set(indices))


def summarize_barrel_detections(all_barrel_detections):
    centers = []

    for frame_entry in all_barrel_detections:
        for barrel in frame_entry["barrels"]:
            centers.append({
                "center_x": barrel["center_x"],
                "center_y": barrel["center_y"],
                "confidence": barrel["confidence"],
                "frame_index": frame_entry["frame_index"],
            })

    if not centers:
        return {
            "detected_frame_count": 0,
            "total_barrel_boxes": 0,
            "average_barrels_per_detected_frame": 0.0,
            "top_barrel_centers": [],
        }

    sorted_centers = sorted(centers, key=lambda c: c["confidence"], reverse=True)
    top_centers = [
        {
            "center_x": c["center_x"],
            "center_y": c["center_y"],
            "confidence": c["confidence"],
            "frame_index": c["frame_index"],
        }
        for c in sorted_centers[:10]
    ]

    detected_frame_count = sum(1 for entry in all_barrel_detections if len(entry["barrels"]) > 0)
    total_barrel_boxes = len(centers)
    average_barrels_per_detected_frame = (
        total_barrel_boxes / detected_frame_count if detected_frame_count > 0 else 0.0
    )

    return {
        "detected_frame_count": detected_frame_count,
        "total_barrel_boxes": total_barrel_boxes,
        "average_barrels_per_detected_frame": round(average_barrels_per_detected_frame, 3),
        "top_barrel_centers": top_centers,
    }


def flatten_barrel_points(all_barrel_detections):
    points = []

    for frame_entry in all_barrel_detections:
        frame_index = frame_entry["frame_index"]
        timestamp_seconds = frame_entry["timestamp_seconds"]
        for barrel in frame_entry["barrels"]:
            points.append({
                "x": float(barrel["center_x"]),
                "y": float(barrel["center_y"]),
                "confidence": float(barrel["confidence"]),
                "frame_index": int(frame_index),
                "timestamp_seconds": timestamp_seconds,
            })

    return points


def initialize_cluster_centers(points, k):
    if not points:
        return []

    sorted_points = sorted(points, key=lambda p: (p["x"], p["y"]))

    if k >= len(sorted_points):
        return [(p["x"], p["y"]) for p in sorted_points[:k]]

    centers = []
    for i in range(k):
        idx = int(round(i * (len(sorted_points) - 1) / max(k - 1, 1)))
        centers.append((sorted_points[idx]["x"], sorted_points[idx]["y"]))

    unique_centers = []
    for cx, cy in centers:
        if not any(distance((cx, cy), existing) < 1.0 for existing in unique_centers):
            unique_centers.append((cx, cy))

    while len(unique_centers) < k:
        fallback = sorted_points[len(unique_centers)]
        unique_centers.append((fallback["x"], fallback["y"]))

    return unique_centers[:k]


def cluster_barrel_points(points, width, height, max_clusters=3, iterations=10):
    if not points:
        return []

    k = min(max_clusters, len(points))
    centers = initialize_cluster_centers(points, k)
    if not centers:
        return []

    diagonal = math.hypot(width, height) if width > 0 and height > 0 else 1000.0
    merge_threshold = clamp(diagonal * 0.08, 40.0, 160.0)

    for _ in range(iterations):
        assignments = [[] for _ in range(len(centers))]

        for point in points:
            distances = [distance((point["x"], point["y"]), center) for center in centers]
            best_index = min(range(len(distances)), key=lambda i: distances[i])
            assignments[best_index].append(point)

        new_centers = []
        for i, assigned_points in enumerate(assignments):
            if not assigned_points:
                new_centers.append(centers[i])
                continue

            weight_sum = sum(max(0.05, p["confidence"]) for p in assigned_points)
            mean_x = sum(p["x"] * max(0.05, p["confidence"]) for p in assigned_points) / weight_sum
            mean_y = sum(p["y"] * max(0.05, p["confidence"]) for p in assigned_points) / weight_sum
            new_centers.append((mean_x, mean_y))

        centers = new_centers

    cluster_infos = []
    for idx, center in enumerate(centers):
        assigned_points = []
        for point in points:
            distances = [distance((point["x"], point["y"]), c) for c in centers]
            best_index = min(range(len(distances)), key=lambda i: distances[i])
            if best_index == idx:
                assigned_points.append(point)

        if not assigned_points:
            continue

        confidence_sum = sum(max(0.05, p["confidence"]) for p in assigned_points)
        avg_confidence = sum(p["confidence"] for p in assigned_points) / len(assigned_points)

        cluster_infos.append({
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "detection_count": int(len(assigned_points)),
            "average_confidence": float(avg_confidence),
            "confidence_sum": float(confidence_sum),
        })

    merged = []
    for cluster in sorted(cluster_infos, key=lambda c: c["confidence_sum"], reverse=True):
        matched = None
        for existing in merged:
            if distance(
                (cluster["center_x"], cluster["center_y"]),
                (existing["center_x"], existing["center_y"])
            ) <= merge_threshold:
                matched = existing
                break

        if matched is None:
            merged.append(cluster)
        else:
            total_weight = matched["confidence_sum"] + cluster["confidence_sum"]
            matched["center_x"] = (
                matched["center_x"] * matched["confidence_sum"] +
                cluster["center_x"] * cluster["confidence_sum"]
            ) / total_weight
            matched["center_y"] = (
                matched["center_y"] * matched["confidence_sum"] +
                cluster["center_y"] * cluster["confidence_sum"]
            ) / total_weight
            matched["detection_count"] += cluster["detection_count"]
            matched["confidence_sum"] += cluster["confidence_sum"]
            matched["average_confidence"] = (
                matched["average_confidence"] + cluster["average_confidence"]
            ) / 2.0

    merged = sorted(merged, key=lambda c: c["detection_count"], reverse=True)[:3]
    return merged


def assign_geometry_barrels_from_clusters(clusters):
    identified = {
        "barrel1": None,  # provisional lower-left
        "barrel2": None,  # provisional lower-right
        "barrel3": None,  # provisional top
    }

    if not clusters:
        return identified, {
            "top": None,
            "lower_left": None,
            "lower_right": None,
        }

    cluster_points = sorted(
        clusters,
        key=lambda c: (float(c["center_y"]), -int(c["detection_count"]))
    )

    top_cluster = cluster_points[0]
    remaining = cluster_points[1:]

    lower_left = None
    lower_right = None

    if len(remaining) == 1:
        only = remaining[0]
        if float(only["center_x"]) <= float(top_cluster["center_x"]):
            lower_left = only
        else:
            lower_right = only
    elif len(remaining) >= 2:
        remaining = sorted(remaining, key=lambda c: float(c["center_x"]))
        lower_left = remaining[0]
        lower_right = remaining[1]

    geometry_named = {
        "top": top_cluster,
        "lower_left": lower_left,
        "lower_right": lower_right,
    }

    if lower_left is not None:
        identified["barrel1"] = {
            "center_x": round(float(lower_left["center_x"]), 2),
            "center_y": round(float(lower_left["center_y"]), 2),
            "detection_count": int(lower_left["detection_count"]),
            "average_confidence": round(float(lower_left["average_confidence"]), 3),
            "geometry_role": "lower_left",
        }

    if lower_right is not None:
        identified["barrel2"] = {
            "center_x": round(float(lower_right["center_x"]), 2),
            "center_y": round(float(lower_right["center_y"]), 2),
            "detection_count": int(lower_right["detection_count"]),
            "average_confidence": round(float(lower_right["average_confidence"]), 3),
            "geometry_role": "lower_right",
        }

    if top_cluster is not None:
        identified["barrel3"] = {
            "center_x": round(float(top_cluster["center_x"]), 2),
            "center_y": round(float(top_cluster["center_y"]), 2),
            "detection_count": int(top_cluster["detection_count"]),
            "average_confidence": round(float(top_cluster["average_confidence"]), 3),
            "geometry_role": "top",
        }

    return identified, geometry_named


def identify_barrels(all_barrel_detections, width, height):
    points = flatten_barrel_points(all_barrel_detections)
    clusters = cluster_barrel_points(points, width, height, max_clusters=3, iterations=10)
    identified, geometry_named = assign_geometry_barrels_from_clusters(clusters)

    geometry_summary = {
        "top": None,
        "lower_left": None,
        "lower_right": None,
    }

    for key, cluster in geometry_named.items():
        if cluster is None:
            continue
        geometry_summary[key] = {
            "center_x": round(float(cluster["center_x"]), 2),
            "center_y": round(float(cluster["center_y"]), 2),
            "detection_count": int(cluster["detection_count"]),
            "average_confidence": round(float(cluster["average_confidence"]), 3),
        }

    return identified, geometry_summary


def build_frame_metrics(sampled_frames, identified_barrels):
    metrics = []

    for frame in sampled_frames:
        horse_detection = frame.get("horse_detection")
        horse_center = None
        nearest_barrel = None
        nearest_distance = None

        dist_map = {
            "barrel1": None,
            "barrel2": None,
            "barrel3": None,
        }

        if horse_detection is not None:
            horse_center = (
                float(horse_detection["tracking_point"][0]),
                float(horse_detection["tracking_point"][1]),
            )

            for barrel_name in BARREL_LABELS:
                barrel_info = identified_barrels.get(barrel_name)
                if barrel_info is None:
                    continue

                barrel_center = (
                    float(barrel_info["center_x"]),
                    float(barrel_info["center_y"]),
                )
                d = distance(horse_center, barrel_center)
                dist_map[barrel_name] = d

            available = [(k, v) for k, v in dist_map.items() if v is not None]
            if available:
                nearest_barrel, nearest_distance = min(available, key=lambda item: item[1])

        metrics.append({
            "frame_index": int(frame["frame_index"]),
            "timestamp_seconds": frame["timestamp_seconds"],
            "horse_detected": horse_detection is not None,
            "horse_center": round_point(horse_center, 2) if horse_center is not None else None,
            "nearest_barrel": nearest_barrel,
            "nearest_barrel_distance_px": round_or_none(nearest_distance, 2),
            "dist_to_barrel1_px": round_or_none(dist_map["barrel1"], 2),
            "dist_to_barrel2_px": round_or_none(dist_map["barrel2"], 2),
            "dist_to_barrel3_px": round_or_none(dist_map["barrel3"], 2),
        })

    return metrics


def angle_degrees(p1, p2):
    if p1 is None or p2 is None:
        return None

    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None

    return math.degrees(math.atan2(dy, dx))


def angle_difference_deg(a, b):
    if a is None or b is None:
        return None
    diff = (b - a + 180.0) % 360.0 - 180.0
    return abs(diff)


def build_motion_samples(frame_metrics):
    valid = [m for m in frame_metrics if m["horse_detected"] and m["horse_center"] is not None]
    motion = []

    for i, metric in enumerate(valid):
        speed_px_per_sec = None
        heading_deg = None

        if i > 0:
            prev = valid[i - 1]
            t1 = prev["timestamp_seconds"]
            t2 = metric["timestamp_seconds"]

            if t1 is not None and t2 is not None:
                dt = float(t2) - float(t1)
                if dt > 1e-9:
                    speed_px_per_sec = distance(prev["horse_center"], metric["horse_center"]) / dt
                    heading_deg = angle_degrees(prev["horse_center"], metric["horse_center"])

        motion.append({
            **metric,
            "speed_px_per_sec": round_or_none(speed_px_per_sec, 2),
            "heading_deg": round_or_none(heading_deg, 2),
        })

    return motion


def find_turn_for_barrel(barrel_name, frame_metrics, width, height):
    valid = [
        m for m in frame_metrics
        if m["horse_detected"]
        and m["horse_center"] is not None
        and m.get(f"dist_to_{barrel_name}_px") is not None
    ]

    if not valid:
        return None

    dist_key = f"dist_to_{barrel_name}_px"
    apex_metric = min(valid, key=lambda m: m[dist_key])
    apex_distance = float(apex_metric[dist_key])

    diagonal = math.hypot(width, height) if width > 0 and height > 0 else 1000.0
    threshold = max(apex_distance * 1.9, apex_distance + 55.0, diagonal * 0.08)

    apex_idx = next(i for i, m in enumerate(valid) if m["frame_index"] == apex_metric["frame_index"])
    start_idx = apex_idx
    end_idx = apex_idx

    while start_idx > 0:
        d = float(valid[start_idx - 1][dist_key])
        if d <= threshold:
            start_idx -= 1
        else:
            break

    while end_idx < len(valid) - 1:
        d = float(valid[end_idx + 1][dist_key])
        if d <= threshold:
            end_idx += 1
        else:
            break

    start_metric = valid[start_idx]
    end_metric = valid[end_idx]

    return {
        "barrel_name": barrel_name,
        "start_frame": int(start_metric["frame_index"]),
        "apex_frame": int(apex_metric["frame_index"]),
        "end_frame": int(end_metric["frame_index"]),
        "start_timestamp_seconds": start_metric["timestamp_seconds"],
        "apex_timestamp_seconds": apex_metric["timestamp_seconds"],
        "end_timestamp_seconds": end_metric["timestamp_seconds"],
        "min_distance_px": round(apex_distance, 2),
    }


def build_turns(frame_metrics, width, height):
    return {
        "barrel1": find_turn_for_barrel("barrel1", frame_metrics, width, height),
        "barrel2": find_turn_for_barrel("barrel2", frame_metrics, width, height),
        "barrel3": find_turn_for_barrel("barrel3", frame_metrics, width, height),
    }


def invert_label_map(actual_to_provisional_map):
    return {v: k for k, v in actual_to_provisional_map.items()}


def remap_barrel_keyed_dict(provisional_dict, actual_to_provisional_map):
    remapped = {}
    for actual_label in BARREL_LABELS:
        provisional_label = actual_to_provisional_map.get(actual_label)
        remapped[actual_label] = provisional_dict.get(provisional_label) if provisional_label else None
    return remapped


def remap_frame_metric_labels(metrics, actual_to_provisional_map):
    provisional_to_actual = invert_label_map(actual_to_provisional_map)
    remapped = []

    for metric in metrics:
        new_metric = dict(metric)

        new_metric["dist_to_barrel1_px"] = metric.get(
            f"dist_to_{actual_to_provisional_map['barrel1']}_px"
        )
        new_metric["dist_to_barrel2_px"] = metric.get(
            f"dist_to_{actual_to_provisional_map['barrel2']}_px"
        )
        new_metric["dist_to_barrel3_px"] = metric.get(
            f"dist_to_{actual_to_provisional_map['barrel3']}_px"
        )

        provisional_nearest = metric.get("nearest_barrel")
        new_metric["nearest_barrel"] = (
            provisional_to_actual.get(provisional_nearest)
            if provisional_nearest is not None
            else None
        )

        remapped.append(new_metric)

    return remapped


def detect_pattern_direction(provisional_turns, provisional_frame_metrics):
    left_turn = provisional_turns.get("barrel1")
    right_turn = provisional_turns.get("barrel2")

    identity_map = {
        "barrel1": "barrel1",
        "barrel2": "barrel2",
        "barrel3": "barrel3",
    }
    right_first_map = {
        "barrel1": "barrel2",
        "barrel2": "barrel1",
        "barrel3": "barrel3",
    }

    if (
        left_turn is not None
        and right_turn is not None
        and left_turn.get("apex_timestamp_seconds") is not None
        and right_turn.get("apex_timestamp_seconds") is not None
    ):
        left_time = float(left_turn["apex_timestamp_seconds"])
        right_time = float(right_turn["apex_timestamp_seconds"])

        if abs(left_time - right_time) > PATTERN_APEX_TIME_TOLERANCE_SECONDS:
            if right_time < left_time:
                return {
                    "pattern_direction": "right",
                    "actual_to_provisional_map": right_first_map,
                    "reason": "right-side lower barrel apex occurred earlier than left-side lower barrel apex",
                    "confidence": 0.95,
                    "method": "turn_apex_timing",
                }
            return {
                "pattern_direction": "left",
                "actual_to_provisional_map": identity_map,
                "reason": "left-side lower barrel apex occurred earlier than right-side lower barrel apex",
                "confidence": 0.95,
                "method": "turn_apex_timing",
            }

    early_candidates = []
    for metric in provisional_frame_metrics:
        if not metric.get("horse_detected"):
            continue
        nearest = metric.get("nearest_barrel")
        if nearest in ("barrel1", "barrel2"):
            early_candidates.append(metric)
        if len(early_candidates) >= 4:
            break

    left_votes = 0
    right_votes = 0

    for metric in early_candidates:
        nearest = metric.get("nearest_barrel")
        if nearest == "barrel1":
            left_votes += 1
        elif nearest == "barrel2":
            right_votes += 1

    if right_votes > left_votes:
        return {
            "pattern_direction": "right",
            "actual_to_provisional_map": right_first_map,
            "reason": "early approach favored the right-side lower barrel",
            "confidence": 0.65,
            "method": "early_approach_vote",
        }

    return {
        "pattern_direction": "left",
        "actual_to_provisional_map": identity_map,
        "reason": "defaulted to left-first",
        "confidence": 0.55,
        "method": "fallback_left",
    }


def compute_path_length(points):
    if len(points) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(points)):
        total += distance(points[i - 1], points[i])

    return total


def average_values(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def get_motion_metric_by_frame(motion_samples, frame_index):
    for m in motion_samples:
        if int(m["frame_index"]) == int(frame_index):
            return m
    return None


def collect_turn_motion_samples(motion_samples, start_frame, end_frame):
    return [
        m for m in motion_samples
        if int(start_frame) <= int(m["frame_index"]) <= int(end_frame)
    ]


def build_barrel_metrics(turns, motion_samples, identified_barrels):
    metrics = {}

    for barrel_name in BARREL_LABELS:
        turn = turns.get(barrel_name)
        barrel_info = identified_barrels.get(barrel_name)

        if turn is None or barrel_info is None:
            metrics[barrel_name] = None
            continue

        turn_samples = collect_turn_motion_samples(
            motion_samples,
            turn["start_frame"],
            turn["end_frame"],
        )

        if not turn_samples:
            metrics[barrel_name] = None
            continue

        apex_sample = get_motion_metric_by_frame(motion_samples, turn["apex_frame"])
        start_sample = get_motion_metric_by_frame(motion_samples, turn["start_frame"])
        end_sample = get_motion_metric_by_frame(motion_samples, turn["end_frame"])

        path_points = [tuple(s["horse_center"]) for s in turn_samples if s["horse_center"] is not None]
        path_length_px = compute_path_length(path_points)

        entry_speeds = [s["speed_px_per_sec"] for s in turn_samples[:2]]
        exit_speeds = [s["speed_px_per_sec"] for s in turn_samples[-2:]]

        entry_speed_avg = average_values(entry_speeds)
        exit_speed_avg = average_values(exit_speeds)

        speed_retention_ratio = None
        if entry_speed_avg is not None and entry_speed_avg > 1e-9 and exit_speed_avg is not None:
            speed_retention_ratio = exit_speed_avg / entry_speed_avg

        heading_start = start_sample["heading_deg"] if start_sample else None
        heading_end = end_sample["heading_deg"] if end_sample else None
        heading_change = angle_difference_deg(heading_start, heading_end)

        metrics[barrel_name] = {
            "start_frame": int(turn["start_frame"]),
            "apex_frame": int(turn["apex_frame"]),
            "end_frame": int(turn["end_frame"]),
            "start_timestamp_seconds": turn["start_timestamp_seconds"],
            "apex_timestamp_seconds": turn["apex_timestamp_seconds"],
            "end_timestamp_seconds": turn["end_timestamp_seconds"],
            "min_distance_px": round_or_none(turn["min_distance_px"], 2),
            "entry_speed_px_per_sec": round_or_none(entry_speed_avg, 2),
            "exit_speed_px_per_sec": round_or_none(exit_speed_avg, 2),
            "speed_retention_ratio": round_or_none(speed_retention_ratio, 3),
            "path_length_px": round_or_none(path_length_px, 2),
            "heading_change_deg": round_or_none(heading_change, 2),
            "barrel_center": round_point((barrel_info["center_x"], barrel_info["center_y"]), 2),
        }

    return metrics


def build_splits(turns, motion_samples):
    valid_motion = [m for m in motion_samples if m["timestamp_seconds"] is not None]
    if not valid_motion:
        return {
            "start_to_barrel1_seconds": None,
            "barrel1_to_barrel2_seconds": None,
            "barrel2_to_barrel3_seconds": None,
            "barrel3_to_home_seconds": None,
        }

    start_time = valid_motion[0]["timestamp_seconds"]
    end_time = valid_motion[-1]["timestamp_seconds"]

    b1 = turns.get("barrel1")
    b2 = turns.get("barrel2")
    b3 = turns.get("barrel3")

    return {
        "start_to_barrel1_seconds": round_or_none(
            (b1["apex_timestamp_seconds"] - start_time)
            if b1 and b1["apex_timestamp_seconds"] is not None
            else None,
            3,
        ),
        "barrel1_to_barrel2_seconds": round_or_none(
            (b2["apex_timestamp_seconds"] - b1["apex_timestamp_seconds"])
            if b1 and b2 and b1["apex_timestamp_seconds"] is not None and b2["apex_timestamp_seconds"] is not None
            else None,
            3,
        ),
        "barrel2_to_barrel3_seconds": round_or_none(
            (b3["apex_timestamp_seconds"] - b2["apex_timestamp_seconds"])
            if b2 and b3 and b2["apex_timestamp_seconds"] is not None and b3["apex_timestamp_seconds"] is not None
            else None,
            3,
        ),
        "barrel3_to_home_seconds": round_or_none(
            (end_time - b3["apex_timestamp_seconds"])
            if b3 and b3["apex_timestamp_seconds"] is not None
            else None,
            3,
        ),
    }


def compute_speed_scores(barrel_metrics):
    scores = {}

    for barrel_name in BARREL_LABELS:
        metric = barrel_metrics.get(barrel_name)
        if metric is None:
            scores[barrel_name] = None
            continue

        entry_speed = metric.get("entry_speed_px_per_sec")
        exit_speed = metric.get("exit_speed_px_per_sec")
        speed_retention = metric.get("speed_retention_ratio")

        if speed_retention is None:
            scores[barrel_name] = None
            continue

        score = 60.0

        score += clamp((speed_retention - 0.85) * 120.0, -25.0, 30.0)

        if entry_speed is not None and exit_speed is not None:
            score += clamp((exit_speed - entry_speed) * 0.05, -15.0, 15.0)

        scores[barrel_name] = round(clamp(score, 0.0, 100.0), 1)

    return scores


def choose_best_barrel(barrel_metrics, speed_scores):
    candidates = []

    for barrel_name, metric in barrel_metrics.items():
        if metric is None:
            continue

        score = 0.0

        min_distance = metric.get("min_distance_px")
        heading_change = metric.get("heading_change_deg")
        speed_retention = metric.get("speed_retention_ratio")
        speed_score = speed_scores.get(barrel_name)

        if min_distance is not None:
            score -= abs(min_distance - 85.0) * 0.18
        if heading_change is not None:
            score += heading_change * 0.28
        if speed_retention is not None:
            score += speed_retention * 14.0
        if speed_score is not None:
            score += speed_score * 0.35

        candidates.append((barrel_name, score))

    if not candidates:
        return None

    return max(candidates, key=lambda x: x[1])[0]


def choose_best_turn(barrel_metrics):
    candidates = []

    for barrel_name, metric in barrel_metrics.items():
        if metric is None:
            continue

        score = 0.0

        path_length = metric.get("path_length_px")
        min_distance = metric.get("min_distance_px")
        heading_change = metric.get("heading_change_deg")
        speed_retention = metric.get("speed_retention_ratio")

        if path_length is not None:
            score -= path_length * 0.02
        if min_distance is not None:
            score -= abs(min_distance - 85.0) * 0.16
        if heading_change is not None:
            score += heading_change * 0.30
        if speed_retention is not None:
            score += speed_retention * 14.0

        candidates.append((barrel_name, score))

    if not candidates:
        return None

    return max(candidates, key=lambda x: x[1])[0]


def build_focus_next(barrel_metrics, speed_scores):
    weakest_barrel = None
    weakest_score = None

    for barrel_name in BARREL_LABELS:
        metric = barrel_metrics.get(barrel_name)
        speed_score = speed_scores.get(barrel_name)

        if metric is None or speed_score is None:
            continue

        min_distance = metric.get("min_distance_px")
        heading_change = metric.get("heading_change_deg")
        speed_retention = metric.get("speed_retention_ratio")

        score = float(speed_score)

        if min_distance is not None:
            score -= abs(min_distance - 85.0) * 0.10
        if heading_change is not None:
            score += max(0.0, min(heading_change, 120.0)) * 0.08
        if speed_retention is not None:
            score += speed_retention * 8.0

        if weakest_score is None or score < weakest_score:
            weakest_score = score
            weakest_barrel = barrel_name

    if weakest_barrel is None:
        return "Keep improving path consistency and turn shape."

    metric = barrel_metrics.get(weakest_barrel)
    if metric is None:
        return f"Focus next on {weakest_barrel.upper()}."

    min_distance = metric.get("min_distance_px")
    speed_retention = metric.get("speed_retention_ratio")

    if min_distance is not None and min_distance > 120:
        return f"Focus next on {weakest_barrel.upper()}: tighten the pocket and shorten the turn."
    if min_distance is not None and min_distance < 50:
        return f"Focus next on {weakest_barrel.upper()}: avoid crowding the barrel and create a cleaner pocket."
    if speed_retention is not None and speed_retention < 0.85:
        return f"Focus next on {weakest_barrel.upper()}: improve exit drive and carry more speed out of the turn."

    return f"Focus next on {weakest_barrel.upper()}: clean up the line and maintain speed through the turn."


def resample_polyline(points, num_samples=120):
    if not points:
        return []
    if len(points) == 1:
        return [points[0] for _ in range(num_samples)]

    points = dedupe_points(points)
    if len(points) == 1:
        return [points[0] for _ in range(num_samples)]

    segment_lengths = []
    cumulative = [0.0]
    total = 0.0

    for i in range(1, len(points)):
        seg_len = distance(points[i - 1], points[i])
        segment_lengths.append(seg_len)
        total += seg_len
        cumulative.append(total)

    if total <= 1e-9:
        return [points[0] for _ in range(num_samples)]

    samples = []
    for i in range(num_samples):
        target = (total * i) / max(num_samples - 1, 1)

        seg_index = 0
        while seg_index < len(segment_lengths) - 1 and cumulative[seg_index + 1] < target:
            seg_index += 1

        start_len = cumulative[seg_index]
        end_len = cumulative[seg_index + 1]
        p1 = points[seg_index]
        p2 = points[seg_index + 1]

        if end_len - start_len <= 1e-9:
            samples.append((float(p1[0]), float(p1[1])))
            continue

        t = (target - start_len) / (end_len - start_len)
        x = float(p1[0]) + t * (float(p2[0]) - float(p1[0]))
        y = float(p1[1]) + t * (float(p2[1]) - float(p1[1]))
        samples.append((x, y))

    return samples


def mirror_points_horiz(points):
    return [(1.0 - float(x), float(y)) for x, y in points]


def build_left_first_ideal_waypoints():
    return [
        (0.50, 1.02),
        (0.44, 0.94),
        (0.34, 0.86),
        (0.24, 0.78),
        (0.14, 0.70),
        (0.10, 0.62),
        (0.14, 0.54),
        (0.24, 0.56),
        (0.36, 0.62),
        (0.48, 0.66),
        (0.62, 0.66),
        (0.76, 0.62),
        (0.88, 0.58),
        (0.92, 0.66),
        (0.88, 0.76),
        (0.76, 0.80),
        (0.64, 0.72),
        (0.56, 0.58),
        (0.52, 0.44),
        (0.50, 0.30),
        (0.48, 0.18),
        (0.42, 0.08),
        (0.56, 0.08),
        (0.60, 0.20),
        (0.56, 0.40),
        (0.50, 0.64),
        (0.46, 0.84),
        (0.44, 1.02),
    ]


def build_ideal_template_path(direction, num_samples=120):
    left_points = build_left_first_ideal_waypoints()

    if direction == "right":
        path = mirror_points_horiz(left_points)
    else:
        path = left_points

    return resample_polyline(path, num_samples=num_samples)


def build_normalized_actual_path(smoothed_points, barrel_geometry):
    if not smoothed_points or not barrel_geometry:
        return {
            "normalized_path": [],
            "transform": None,
        }

    top = barrel_geometry.get("top")
    lower_left = barrel_geometry.get("lower_left")
    lower_right = barrel_geometry.get("lower_right")

    if top is None or lower_left is None or lower_right is None:
        return {
            "normalized_path": [],
            "transform": None,
        }

    left_x = float(lower_left["center_x"])
    right_x = float(lower_right["center_x"])
    top_y = float(top["center_y"])

    if abs(right_x - left_x) < 1e-6:
        return {
            "normalized_path": [],
            "transform": None,
        }

    path_y_values = [float(p[1]) for p in smoothed_points]
    max_path_y = max(path_y_values) if path_y_values else top_y + 1.0

    if max_path_y <= top_y:
        max_path_y = top_y + 1.0

    normalized = []
    for x, y in smoothed_points:
        nx = CANONICAL_LEFT_BARREL_X + (
            (float(x) - left_x) / (right_x - left_x)
        ) * (CANONICAL_RIGHT_BARREL_X - CANONICAL_LEFT_BARREL_X)

        ny = CANONICAL_TOP_BARREL_Y + (
            (float(y) - top_y) / max(max_path_y - top_y, 1e-9)
        ) * (CANONICAL_HOME_Y - CANONICAL_TOP_BARREL_Y)

        normalized.append((float(nx), float(ny)))

    return {
        "normalized_path": normalized,
        "transform": {
            "left_x": round(left_x, 3),
            "right_x": round(right_x, 3),
            "top_y": round(top_y, 3),
            "max_path_y": round(max_path_y, 3),
        },
    }


def compare_paths(actual_path, ideal_path):
    if not actual_path or not ideal_path or len(actual_path) < 4 or len(ideal_path) < 4:
        return None

    sample_count = 100
    actual_resampled = resample_polyline(actual_path, sample_count)
    ideal_resampled = resample_polyline(ideal_path, sample_count)

    pointwise_distances = [
        distance(actual_resampled[i], ideal_resampled[i])
        for i in range(sample_count)
    ]

    mean_distance = sum(pointwise_distances) / len(pointwise_distances) if pointwise_distances else None
    max_distance = max(pointwise_distances) if pointwise_distances else None

    return {
        "sample_count": sample_count,
        "mean_distance": round_or_none(mean_distance, 4),
        "max_distance": round_or_none(max_distance, 4),
        "actual_resampled_path": [round_point(p, 4) for p in actual_resampled],
        "ideal_resampled_path": [round_point(p, 4) for p in ideal_resampled],
    }


def draw_overlay(frame, horse_detection, barrel_detections, identified_barrels, frame_index, timestamp_seconds):
    overlay = frame.copy()

    for barrel in barrel_detections or []:
        x1 = int(barrel["x1"])
        y1 = int(barrel["y1"])
        x2 = int(barrel["x2"])
        y2 = int(barrel["y2"])
        conf = barrel.get("confidence")

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            overlay,
            f"barrel {conf}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

    if identified_barrels:
        for barrel_name, barrel_info in identified_barrels.items():
            if barrel_info is None:
                continue
            x = int(barrel_info["center_x"])
            y = int(barrel_info["center_y"])
            cv2.circle(overlay, (x, y), 12, (255, 255, 0), 2)
            cv2.putText(
                overlay,
                barrel_name.upper(),
                (x + 8, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    if horse_detection:
        x1, y1, x2, y2 = [int(v) for v in horse_detection["bbox"]]
        cx, cy = [int(v) for v in horse_detection["tracking_point"]]
        conf = horse_detection.get("confidence")

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            f"horse {conf}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    header_1 = f"frame {frame_index}"
    header_2 = f"time {timestamp_seconds}s" if timestamp_seconds is not None else "time unknown"

    cv2.putText(
        overlay,
        header_1,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        header_2,
        (16, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay


def save_path_map(
    width,
    height,
    raw_points,
    accepted_points,
    smooth_points_list,
    out_path,
    identified_barrels=None,
    normalized_actual_path=None,
    template_path=None,
):
    canvas_width = min(width, 960)
    canvas_height = min(height, 540)

    scale_x = canvas_width / float(max(width, 1))
    scale_y = canvas_height / float(max(height, 1))

    canvas = 255 * (cv2.UMat(canvas_height, canvas_width, cv2.CV_8UC3).get() * 0 + 1)

    for pt in raw_points:
        cv2.circle(
            canvas,
            (int(pt[0] * scale_x), int(pt[1] * scale_y)),
            2,
            (190, 190, 190),
            -1,
        )

    for pt in accepted_points:
        cv2.circle(
            canvas,
            (int(pt[0] * scale_x), int(pt[1] * scale_y)),
            3,
            (0, 165, 255),
            -1,
        )

    for i in range(1, len(smooth_points_list)):
        p1 = (int(smooth_points_list[i - 1][0] * scale_x), int(smooth_points_list[i - 1][1] * scale_y))
        p2 = (int(smooth_points_list[i][0] * scale_x), int(smooth_points_list[i][1] * scale_y))
        cv2.line(canvas, p1, p2, (255, 0, 0), 3)

    for pt in smooth_points_list:
        cv2.circle(
            canvas,
            (int(pt[0] * scale_x), int(pt[1] * scale_y)),
            4,
            (0, 0, 255),
            -1,
        )

    if identified_barrels:
        for barrel_name, barrel_info in identified_barrels.items():
            if barrel_info is None:
                continue
            x = int(barrel_info["center_x"] * scale_x)
            y = int(barrel_info["center_y"] * scale_y)
            cv2.circle(canvas, (x, y), 12, (0, 255, 255), 2)
            cv2.putText(
                canvas,
                barrel_name.upper(),
                (x + 8, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 120, 255),
                2,
                cv2.LINE_AA,
            )

    if normalized_actual_path and template_path:
        mini_w = 260
        mini_h = 320
        origin_x = max(20, canvas_width - mini_w - 20)
        origin_y = 20

        cv2.rectangle(
            canvas,
            (origin_x, origin_y),
            (origin_x + mini_w, origin_y + mini_h),
            (230, 230, 230),
            -1,
        )
        cv2.rectangle(
            canvas,
            (origin_x, origin_y),
            (origin_x + mini_w, origin_y + mini_h),
            (120, 120, 120),
            2,
        )

        def draw_norm_polyline(points, color, thickness):
            if not points or len(points) < 2:
                return
            for i in range(1, len(points)):
                x1 = origin_x + int(points[i - 1][0] * mini_w)
                y1 = origin_y + int(points[i - 1][1] * mini_h)
                x2 = origin_x + int(points[i][0] * mini_w)
                y2 = origin_y + int(points[i][1] * mini_h)
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)

        draw_norm_polyline(template_path, (185, 185, 185), 2)
        draw_norm_polyline(normalized_actual_path, (0, 140, 255), 2)

        canonical_barrels = [
            ("B1", CANONICAL_LEFT_BARREL_X, CANONICAL_LOWER_BARREL_Y),
            ("B2", CANONICAL_RIGHT_BARREL_X, CANONICAL_LOWER_BARREL_Y),
            ("B3", CANONICAL_TOP_BARREL_X, CANONICAL_TOP_BARREL_Y),
        ]
        for label, bx, by in canonical_barrels:
            px = origin_x + int(bx * mini_w)
            py = origin_y + int(by * mini_h)
            cv2.circle(canvas, (px, py), 8, (0, 0, 0), 2)
            cv2.putText(
                canvas,
                label,
                (px + 8, max(origin_y + 16, py - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            "ideal vs actual",
            (origin_x + 10, origin_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

    safe_imwrite(out_path, canvas)


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else None

    if not video_path:
        fail("No video path was provided.")
        return

    if not os.path.exists(video_path):
        fail("Video file does not exist.", {"video_path": video_path})
        return

    if not os.path.exists(BARREL_MODEL_PATH):
        fail("Barrel model file does not exist.", {"barrel_model_path": BARREL_MODEL_PATH})
        return

    if not os.path.exists(HORSE_MODEL_PATH):
        fail("Horse model file does not exist.", {"horse_model_path": HORSE_MODEL_PATH})
        return

    output_dir = f"{video_path}_frames"
    os.makedirs(output_dir, exist_ok=True)

    try:
        log_err("Loading models...")
        barrel_model = load_model(BARREL_MODEL_PATH)
        horse_model = load_model(HORSE_MODEL_PATH)
    except Exception as model_error:
        fail("Failed to load YOLO model.", {"details": str(model_error)})
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        fail("OpenCV could not open the video.", {"video_path": video_path})
        return

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0

        if frame_count <= 0 or original_width <= 0 or original_height <= 0:
            fail(
                "Video metadata could not be read correctly.",
                {
                    "video_path": video_path,
                    "frame_count": frame_count,
                    "fps": fps,
                    "width": original_width,
                    "height": original_height,
                },
            )
            return

        log_err(
            f"Video opened. frame_count={frame_count}, fps={round(fps, 3)}, "
            f"width={original_width}, height={original_height}, duration={round(duration, 3)}"
        )

        sample_indices = build_sample_frame_indices(frame_count, fps)
        max_jump = adaptive_max_jump(original_width, original_height)

        sampled_frames = []
        all_barrel_detections = []

        horse_detected_count = 0
        read_success_count = 0
        rejected_jump_count = 0
        missed_detection_count = 0

        raw_trajectory_points = []
        accepted_points = []
        track_points = []

        previous_accepted_point = None

        for idx, target_frame in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()

            image_path = None
            overlay_image_path = None
            horse_detection = None
            rejection_reason = None
            barrel_detections = []

            percent = (target_frame / max(frame_count - 1, 1)) if frame_count > 1 else 0.0
            timestamp_seconds = round(target_frame / fps, 3) if fps > 0 else None

            chosen_point_for_track = None

            if ret and frame is not None:
                read_success_count += 1

                inference_frame, x_ratio, y_ratio = resize_for_inference(frame)

                # Run barrel detection only on every other sampled frame to reduce load
                if idx % 2 == 0:
                    barrel_detections_resized = detect_barrels_in_frame(
                        inference_frame,
                        barrel_model,
                        confidence_threshold=BARREL_CONFIDENCE_THRESHOLD,
                    )
                    barrel_detections = scale_barrel_detections_to_original(
                        barrel_detections_resized,
                        x_ratio,
                        y_ratio,
                    )

                all_barrel_detections.append({
                    "frame_index": int(target_frame),
                    "timestamp_seconds": timestamp_seconds,
                    "barrels": barrel_detections,
                })

                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    horse_results = horse_model.predict(
                        source=inference_frame,
                        conf=HORSE_CONFIDENCE_THRESHOLD,
                        classes=[HORSE_CLASS_ID],
                        verbose=False,
                    )

                candidates = []
                if horse_results and len(horse_results) > 0:
                    candidates = build_horse_candidates(horse_results[0], x_ratio=x_ratio, y_ratio=y_ratio)

                best_candidate = choose_best_candidate(
                    candidates,
                    previous_accepted_point,
                    original_width,
                    original_height,
                )

                if best_candidate is not None:
                    horse_detected_count += 1
                    horse_detection = {
                        "confidence": best_candidate["confidence"],
                        "bbox": best_candidate["bbox"],
                        "center": best_candidate["center"],
                        "tracking_point": round_point(best_candidate["tracking_point"], 2),
                    }

                    current_point = best_candidate["tracking_point"]
                    raw_trajectory_points.append(current_point)

                    if previous_accepted_point is None:
                        accepted_points.append(current_point)
                        chosen_point_for_track = current_point
                        previous_accepted_point = current_point
                    else:
                        jump_distance = distance(previous_accepted_point, current_point)

                        if jump_distance <= max_jump:
                            accepted_points.append(current_point)
                            chosen_point_for_track = current_point
                            previous_accepted_point = current_point
                        else:
                            rejected_jump_count += 1
                            rejection_reason = f"jump_rejected_{round(jump_distance, 2)}px"
                else:
                    missed_detection_count += 1

            track_points.append(chosen_point_for_track)

            sampled_frames.append({
                "percent": round(percent, 4),
                "frame_index": int(target_frame),
                "timestamp_seconds": timestamp_seconds,
                "read_success": bool(ret and frame is not None),
                "image_path": image_path,
                "overlay_image_path": overlay_image_path,
                "horse_detection": horse_detection,
                "barrel_detections": barrel_detections,
                "barrel_detection_count": len(barrel_detections),
                "rejection_reason": rejection_reason,
            })

        interpolated_track = interpolate_small_gaps(track_points, max_gap=INTERPOLATION_MAX_GAP)
        smoothed_points = [p for p in interpolated_track if p is not None]
        smoothed_points = smooth_points(smoothed_points, alpha=SMOOTHING_ALPHA)
        smoothed_points = dedupe_points(smoothed_points)

        barrel_detection_summary = summarize_barrel_detections(all_barrel_detections)

        provisional_identified_barrels, barrel_geometry = identify_barrels(
            all_barrel_detections,
            original_width,
            original_height,
        )

        provisional_frame_metrics = build_frame_metrics(sampled_frames, provisional_identified_barrels)
        provisional_motion_samples = build_motion_samples(provisional_frame_metrics)
        provisional_turns = build_turns(provisional_frame_metrics, original_width, original_height)

        pattern_direction_info = detect_pattern_direction(provisional_turns, provisional_frame_metrics)
        actual_to_provisional_map = pattern_direction_info["actual_to_provisional_map"]

        identified_barrels = remap_barrel_keyed_dict(provisional_identified_barrels, actual_to_provisional_map)
        frame_metrics = remap_frame_metric_labels(provisional_frame_metrics, actual_to_provisional_map)
        motion_samples = build_motion_samples(frame_metrics)
        turns = remap_barrel_keyed_dict(provisional_turns, actual_to_provisional_map)

        barrel_metrics = build_barrel_metrics(turns, motion_samples, identified_barrels)
        splits = build_splits(turns, motion_samples)
        speed_scores = compute_speed_scores(barrel_metrics)
        best_barrel = choose_best_barrel(barrel_metrics, speed_scores)
        best_turn = choose_best_turn(barrel_metrics)
        focus_next = build_focus_next(barrel_metrics, speed_scores)

        tracking_quality = {
            "sampled_frame_count": len(sample_indices),
            "read_success_count": read_success_count,
            "horse_detected_count": horse_detected_count,
            "missed_detection_count": missed_detection_count,
            "rejected_jump_count": rejected_jump_count,
            "max_jump_threshold_px": round(float(max_jump), 2),
            "read_success_rate": round(read_success_count / max(len(sample_indices), 1), 4),
            "horse_detection_rate": round(horse_detected_count / max(read_success_count, 1), 4),
            "accepted_point_rate": round(len(accepted_points) / max(horse_detected_count, 1), 4),
            "interpolated_path_point_count": len(smoothed_points),
        }

        direction = pattern_direction_info.get("pattern_direction") or "left"
        ideal_template_path = build_ideal_template_path(direction, num_samples=120)

        normalized_actual_path_result = build_normalized_actual_path(smoothed_points, barrel_geometry)
        normalized_actual_path = normalized_actual_path_result["normalized_path"]
        template_path_comparison = compare_paths(normalized_actual_path, ideal_template_path)

        path_map_path = None
        if len(smoothed_points) > 1:
            path_map_path = f"{video_path}_path_map.jpg"
            save_path_map(
                original_width,
                original_height,
                raw_trajectory_points,
                accepted_points,
                smoothed_points,
                path_map_path,
                identified_barrels=identified_barrels,
                normalized_actual_path=template_path_comparison["actual_resampled_path"] if template_path_comparison else None,
                template_path=template_path_comparison["ideal_resampled_path"] if template_path_comparison else None,
            )

        smoothed_path_points = [round_point(pt, 2) for pt in smoothed_points]
        normalized_smoothed_path_points = normalize_points_to_unit_box(smoothed_points)

        output = {
            "ok": True,
            "message": "Lean MVP barrel path analysis completed.",
            "video_path": video_path,
            "frame_count": frame_count,
            "fps": round(fps, 3),
            "width": original_width,
            "height": original_height,
            "duration_seconds": round(duration, 3),

            "horse_detected_frames": horse_detected_count,
            "raw_trajectory_point_count": len(raw_trajectory_points),
            "accepted_trajectory_point_count": len(accepted_points),
            "smoothed_trajectory_point_count": len(smoothed_points),

            "smoothed_path_points": smoothed_path_points,
            "normalized_smoothed_path_points": normalized_smoothed_path_points,
            "path_map_path": path_map_path,

            "tracking_quality": tracking_quality,
            "barrel_detection_summary": barrel_detection_summary,
            "barrel_geometry": barrel_geometry,

            "pattern_direction": direction,
            "pattern_direction_info": pattern_direction_info,
            "identified_barrels": identified_barrels,
            "turns": turns,
            "splits": splits,
            "barrel_metrics": barrel_metrics,

            "normalized_actual_path_transform": normalized_actual_path_result["transform"],
            "normalized_actual_template_path": [round_point(p, 4) for p in normalized_actual_path],
            "ideal_template_path": [round_point(p, 4) for p in ideal_template_path],
            "template_path_comparison": template_path_comparison,

            "speed_scores": speed_scores,
            "highlights": {
                "best_barrel": best_barrel,
                "best_turn": best_turn,
                "focus_next": focus_next,
            },

            "frame_metrics": frame_metrics,
            "motion_samples": motion_samples,
            "sampled_frames": sampled_frames,
        }

        emit_json(output)

    except Exception as runtime_error:
        log_err("Python analysis crashed:", str(runtime_error))
        log_err(traceback.format_exc())

        emit_json({
            "ok": False,
            "error": "Python analysis crashed",
            "details": str(runtime_error),
            "traceback": traceback.format_exc(),
            "video_path": video_path,
        })
    finally:
        cap.release()


if __name__ == "__main__":
    main()