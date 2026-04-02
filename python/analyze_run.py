import os
import sys
import json
import math
import cv2
import contextlib
import io
import traceback

# Must be set before ultralytics import
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

# Tuned lean settings
HORSE_CONFIDENCE_THRESHOLD = 0.18
BARREL_CONFIDENCE_THRESHOLD = 0.48

TARGET_SAMPLE_FPS = 2.0
MIN_SAMPLED_FRAMES = 12
MAX_SAMPLED_FRAMES = 20

MAX_INFERENCE_WIDTH = 576
MAX_INFERENCE_HEIGHT = 324

SMOOTHING_ALPHA = 0.34
INTERPOLATION_MAX_GAP = 3

CANONICAL_LEFT_BARREL_X = 0.24
CANONICAL_RIGHT_BARREL_X = 0.76
CANONICAL_TOP_BARREL_X = 0.50

CANONICAL_TOP_BARREL_Y = 0.22
CANONICAL_LOWER_BARREL_Y = 0.68
CANONICAL_HOME_Y = 0.94


def log_err(*args):
    print(*args, file=sys.stderr, flush=True)


def emit_json(payload):
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def fail(message, extra=None):
    payload = {"ok": False, "error": message}
    if extra and isinstance(extra, dict):
        payload.update(extra)
    emit_json(payload)


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def round_or_none(value, decimals=2):
    if value is None:
        return None
    return round(float(value), decimals)


def round_point(point, decimals=2):
    if point is None:
        return None
    return [round(float(point[0]), decimals), round(float(point[1]), decimals)]


def distance(p1, p2):
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


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

    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame, 1.0, 1.0

    width_scale = MAX_INFERENCE_WIDTH / float(w)
    height_scale = MAX_INFERENCE_HEIGHT / float(h)
    scale = min(width_scale, height_scale, 1.0)

    if scale >= 1.0:
        return frame, 1.0, 1.0

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_ratio = w / float(new_w)
    y_ratio = h / float(new_h)

    return resized, x_ratio, y_ratio


def compute_bottom_center(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = y2
    return cx, cy


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

    if len(indices) < MIN_SAMPLED_FRAMES and frame_count > MIN_SAMPLED_FRAMES:
        desired = min(MIN_SAMPLED_FRAMES, frame_count)
        indices = sorted(
            set(int(round(i * (frame_count - 1) / max(desired - 1, 1))) for i in range(desired))
        )

    if len(indices) > MAX_SAMPLED_FRAMES:
        desired = MAX_SAMPLED_FRAMES
        indices = sorted(
            set(int(round(i * (frame_count - 1) / max(desired - 1, 1))) for i in range(desired))
        )

    return sorted(set(indices))


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
        h = max(0.0, y2 - y1)

        tracking_point = (float(cx), float(y2 - h * 0.10))

        candidates.append(
            {
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
            }
        )

    return candidates


def choose_best_horse_candidate(candidates, prev_point, frame_width, frame_height):
    if not candidates:
        return None

    if prev_point is None:
        return max(candidates, key=lambda c: (c["confidence"], c["area"]))

    diagonal = math.hypot(frame_width, frame_height) if frame_width > 0 and frame_height > 0 else 1.0

    best = None
    best_score = None

    for c in candidates:
        dist = distance(prev_point, c["tracking_point"])
        normalized_dist = dist / diagonal
        score = (c["confidence"] * 2.15) + min(c["area"] / 45000.0, 1.1) - (normalized_dist * 2.1)

        if best_score is None or score > best_score:
            best_score = score
            best = c

    return best


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

        barrels.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": round(conf, 3),
                "center_x": float((x1 + x2) / 2.0),
                "center_y": float((y1 + y2) / 2.0),
            }
        )

    return barrels


def scale_barrels_to_original(barrels, x_ratio, y_ratio):
    scaled = []
    for barrel in barrels:
        scaled.append(
            {
                "x1": int(round(float(barrel["x1"]) * x_ratio)),
                "y1": int(round(float(barrel["y1"]) * y_ratio)),
                "x2": int(round(float(barrel["x2"]) * x_ratio)),
                "y2": int(round(float(barrel["y2"]) * y_ratio)),
                "confidence": barrel["confidence"],
                "center_x": int(round(float(barrel["center_x"]) * x_ratio)),
                "center_y": int(round(float(barrel["center_y"]) * y_ratio)),
            }
        )
    return scaled


def adaptive_max_jump(width, height):
    diagonal = math.hypot(width, height) if width > 0 and height > 0 else 1000.0
    return clamp(diagonal * 0.19, 150.0, 420.0)


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


def dedupe_points(points, min_dist=6.0):
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


def average_values(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def identify_barrels_simple(all_barrel_detections):
    points = []

    for frame_entry in all_barrel_detections:
        for barrel in frame_entry["barrels"]:
            points.append(
                {
                    "x": float(barrel["center_x"]),
                    "y": float(barrel["center_y"]),
                    "confidence": float(barrel["confidence"]),
                }
            )

    result = {
        "barrel1": None,  # lower-left provisional
        "barrel2": None,  # lower-right provisional
        "barrel3": None,  # top provisional
    }

    geometry = {
        "top": None,
        "lower_left": None,
        "lower_right": None,
    }

    if len(points) < 3:
        return result, geometry

    weighted_top_candidates = sorted(points, key=lambda p: (p["y"], -p["confidence"]))
    top_candidates = weighted_top_candidates[: max(3, min(6, len(weighted_top_candidates)))]
    top_x = average_values([p["x"] for p in top_candidates])
    top_y = average_values([p["y"] for p in top_candidates])

    lower_candidates = [p for p in points if p["y"] > top_y + 18]
    if len(lower_candidates) < 2:
      lower_candidates = [p for p in points if p not in top_candidates]

    if len(lower_candidates) < 2:
        return result, geometry

    sorted_lower = sorted(lower_candidates, key=lambda p: p["x"])
    split_index = max(1, len(sorted_lower) // 2)

    left_group = sorted_lower[:split_index]
    right_group = sorted_lower[split_index:]

    if not left_group or not right_group:
        median_x = average_values([p["x"] for p in sorted_lower])
        left_group = [p for p in sorted_lower if p["x"] <= median_x]
        right_group = [p for p in sorted_lower if p["x"] > median_x]

    if not left_group or not right_group:
        return result, geometry

    lower_left = {
        "center_x": average_values([p["x"] for p in left_group]),
        "center_y": average_values([p["y"] for p in left_group]),
        "detection_count": len(left_group),
        "average_confidence": average_values([p["confidence"] for p in left_group]),
    }
    lower_right = {
        "center_x": average_values([p["x"] for p in right_group]),
        "center_y": average_values([p["y"] for p in right_group]),
        "detection_count": len(right_group),
        "average_confidence": average_values([p["confidence"] for p in right_group]),
    }
    top_cluster = {
        "center_x": top_x,
        "center_y": top_y,
        "detection_count": len(top_candidates),
        "average_confidence": average_values([p["confidence"] for p in top_candidates]),
    }

    geometry["top"] = {
        "center_x": round(top_cluster["center_x"], 2),
        "center_y": round(top_cluster["center_y"], 2),
        "detection_count": int(top_cluster["detection_count"]),
        "average_confidence": round(top_cluster["average_confidence"], 3),
    }
    geometry["lower_left"] = {
        "center_x": round(lower_left["center_x"], 2),
        "center_y": round(lower_left["center_y"], 2),
        "detection_count": int(lower_left["detection_count"]),
        "average_confidence": round(lower_left["average_confidence"], 3),
    }
    geometry["lower_right"] = {
        "center_x": round(lower_right["center_x"], 2),
        "center_y": round(lower_right["center_y"], 2),
        "detection_count": int(lower_right["detection_count"]),
        "average_confidence": round(lower_right["average_confidence"], 3),
    }

    result["barrel1"] = {
        "center_x": round(lower_left["center_x"], 2),
        "center_y": round(lower_left["center_y"], 2),
        "detection_count": int(lower_left["detection_count"]),
        "average_confidence": round(lower_left["average_confidence"], 3),
        "geometry_role": "lower_left",
    }
    result["barrel2"] = {
        "center_x": round(lower_right["center_x"], 2),
        "center_y": round(lower_right["center_y"], 2),
        "detection_count": int(lower_right["detection_count"]),
        "average_confidence": round(lower_right["average_confidence"], 3),
        "geometry_role": "lower_right",
    }
    result["barrel3"] = {
        "center_x": round(top_cluster["center_x"], 2),
        "center_y": round(top_cluster["center_y"], 2),
        "detection_count": int(top_cluster["detection_count"]),
        "average_confidence": round(top_cluster["average_confidence"], 3),
        "geometry_role": "top",
    }

    return result, geometry


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

            for barrel_name in ("barrel1", "barrel2", "barrel3"):
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

        metrics.append(
            {
                "frame_index": int(frame["frame_index"]),
                "timestamp_seconds": frame["timestamp_seconds"],
                "horse_detected": horse_detection is not None,
                "horse_center": round_point(horse_center, 2) if horse_center is not None else None,
                "nearest_barrel": nearest_barrel,
                "nearest_barrel_distance_px": round_or_none(nearest_distance, 2),
                "dist_to_barrel1_px": round_or_none(dist_map["barrel1"], 2),
                "dist_to_barrel2_px": round_or_none(dist_map["barrel2"], 2),
                "dist_to_barrel3_px": round_or_none(dist_map["barrel3"], 2),
            }
        )

    return metrics


def detect_pattern_direction(frame_metrics):
    lower_frames = []

    for metric in frame_metrics:
        if not metric["horse_detected"]:
            continue
        if metric["nearest_barrel"] in ("barrel1", "barrel2"):
            lower_frames.append(metric)

    if not lower_frames:
        return {
            "pattern_direction": "left",
            "actual_to_provisional_map": {
                "barrel1": "barrel1",
                "barrel2": "barrel2",
                "barrel3": "barrel3",
            },
            "reason": "defaulted to left-first because early lower-barrel approach was unclear",
            "confidence": 0.5,
            "method": "fallback_left",
        }

    vote_window = lower_frames[: min(4, len(lower_frames))]
    left_votes = sum(1 for m in vote_window if m["nearest_barrel"] == "barrel1")
    right_votes = sum(1 for m in vote_window if m["nearest_barrel"] == "barrel2")

    if right_votes > left_votes:
        return {
            "pattern_direction": "right",
            "actual_to_provisional_map": {
                "barrel1": "barrel2",
                "barrel2": "barrel1",
                "barrel3": "barrel3",
            },
            "reason": "early lower-barrel approach favored the right side",
            "confidence": 0.78,
            "method": "early_lower_votes",
        }

    return {
        "pattern_direction": "left",
        "actual_to_provisional_map": {
            "barrel1": "barrel1",
            "barrel2": "barrel2",
            "barrel3": "barrel3",
        },
        "reason": "early lower-barrel approach favored the left side",
        "confidence": 0.78,
        "method": "early_lower_votes",
    }


def invert_label_map(actual_to_provisional_map):
    return {v: k for k, v in actual_to_provisional_map.items()}


def remap_barrel_keyed_dict(provisional_dict, actual_to_provisional_map):
    remapped = {}
    for actual_label in ("barrel1", "barrel2", "barrel3"):
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


def find_barrel_apex(barrel_name, frame_metrics):
    valid = [
        m
        for m in frame_metrics
        if m["horse_detected"] and m.get(f"dist_to_{barrel_name}_px") is not None
    ]

    if not valid:
        return None

    dist_key = f"dist_to_{barrel_name}_px"
    ordered = sorted(valid, key=lambda m: m[dist_key])
    best_candidates = ordered[: min(3, len(ordered))]
    apex_metric = min(best_candidates, key=lambda m: m["frame_index"])

    return {
        "barrel_name": barrel_name,
        "start_frame": int(apex_metric["frame_index"]),
        "apex_frame": int(apex_metric["frame_index"]),
        "end_frame": int(apex_metric["frame_index"]),
        "start_timestamp_seconds": apex_metric["timestamp_seconds"],
        "apex_timestamp_seconds": apex_metric["timestamp_seconds"],
        "end_timestamp_seconds": apex_metric["timestamp_seconds"],
        "min_distance_px": round_or_none(apex_metric[dist_key], 2),
    }


def build_turns(frame_metrics):
    return {
        "barrel1": find_barrel_apex("barrel1", frame_metrics),
        "barrel2": find_barrel_apex("barrel2", frame_metrics),
        "barrel3": find_barrel_apex("barrel3", frame_metrics),
    }


def enforce_turn_order(turns, direction):
    b1 = turns.get("barrel1")
    b2 = turns.get("barrel2")
    b3 = turns.get("barrel3")

    ordered = []
    for label, turn in [("barrel1", b1), ("barrel2", b2), ("barrel3", b3)]:
        if turn and turn.get("apex_timestamp_seconds") is not None:
            ordered.append((label, float(turn["apex_timestamp_seconds"])))

    if len(ordered) < 2:
        return turns

    # Expected logical order is actual barrel1 -> actual barrel2 -> actual barrel3
    # If timings violate that badly, null out later unreliable splits rather than returning nonsense
    t1 = b1["apex_timestamp_seconds"] if b1 else None
    t2 = b2["apex_timestamp_seconds"] if b2 else None
    t3 = b3["apex_timestamp_seconds"] if b3 else None

    fixed = dict(turns)

    if t1 is not None and t2 is not None and t2 <= t1:
        fixed["barrel2"] = None
    if fixed.get("barrel2") is not None:
        t2 = fixed["barrel2"]["apex_timestamp_seconds"]
    if t2 is not None and t3 is not None and t3 <= t2:
        fixed["barrel3"] = None

    return fixed


def build_splits(turns, frame_metrics):
    valid = [m for m in frame_metrics if m["horse_detected"] and m["timestamp_seconds"] is not None]
    if not valid:
        return {
            "start_to_barrel1_seconds": None,
            "barrel1_to_barrel2_seconds": None,
            "barrel2_to_barrel3_seconds": None,
            "barrel3_to_home_seconds": None,
        }

    start_time = valid[0]["timestamp_seconds"]
    end_time = valid[-1]["timestamp_seconds"]

    b1 = turns.get("barrel1")
    b2 = turns.get("barrel2")
    b3 = turns.get("barrel3")

    s1 = (
        b1["apex_timestamp_seconds"] - start_time
        if b1 and b1["apex_timestamp_seconds"] is not None
        else None
    )
    s2 = (
        b2["apex_timestamp_seconds"] - b1["apex_timestamp_seconds"]
        if b1 and b2 and b1["apex_timestamp_seconds"] is not None and b2["apex_timestamp_seconds"] is not None
        else None
    )
    s3 = (
        b3["apex_timestamp_seconds"] - b2["apex_timestamp_seconds"]
        if b2 and b3 and b2["apex_timestamp_seconds"] is not None and b3["apex_timestamp_seconds"] is not None
        else None
    )
    s4 = (
        end_time - b3["apex_timestamp_seconds"]
        if b3 and b3["apex_timestamp_seconds"] is not None
        else None
    )

    def safe_split(value):
        if value is None:
            return None
        if value < 0 or value > 60:
            return None
        return round_or_none(value, 3)

    return {
        "start_to_barrel1_seconds": safe_split(s1),
        "barrel1_to_barrel2_seconds": safe_split(s2),
        "barrel2_to_barrel3_seconds": safe_split(s3),
        "barrel3_to_home_seconds": safe_split(s4),
    }


def build_highlights(frame_metrics):
    barrel_distances = {
        "barrel1": [],
        "barrel2": [],
        "barrel3": [],
    }

    for metric in frame_metrics:
        for barrel_name in ("barrel1", "barrel2", "barrel3"):
            d = metric.get(f"dist_to_{barrel_name}_px")
            if d is not None:
                barrel_distances[barrel_name].append(float(d))

    avg_distances = {
        k: (sum(v) / len(v) if v else None)
        for k, v in barrel_distances.items()
    }

    available = [(k, v) for k, v in avg_distances.items() if v is not None]
    best_barrel = None
    if available:
        best_barrel = min(available, key=lambda item: item[1])[0]

    best_turn = best_barrel

    focus_next = "Clean up the line and stay tighter around the weakest barrel."

    if available:
        weakest_barrel = max(available, key=lambda item: item[1])[0]
        if weakest_barrel == "barrel1":
            focus_next = "Cleaner entry to 1st barrel"
        elif weakest_barrel == "barrel2":
            focus_next = "Stay more efficient between the 1st and 2nd barrels."
        elif weakest_barrel == "barrel3":
            focus_next = "Carry a cleaner line into and out of the 3rd barrel."

    return {
        "best_barrel": best_barrel,
        "best_turn": best_turn,
        "focus_next": focus_next,
    }


def summarize_barrel_detections(all_barrel_detections):
    centers = []

    for frame_entry in all_barrel_detections:
        for barrel in frame_entry["barrels"]:
            centers.append(
                {
                    "center_x": barrel["center_x"],
                    "center_y": barrel["center_y"],
                    "confidence": barrel["confidence"],
                    "frame_index": frame_entry["frame_index"],
                }
            )

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
        for c in sorted_centers[:8]
    ]

    detected_frame_count = sum(1 for entry in all_barrel_detections if len(entry["barrels"]) > 0)
    total_barrel_boxes = len(centers)
    avg = total_barrel_boxes / detected_frame_count if detected_frame_count > 0 else 0.0

    return {
        "detected_frame_count": detected_frame_count,
        "total_barrel_boxes": total_barrel_boxes,
        "average_barrels_per_detected_frame": round(avg, 3),
        "top_barrel_centers": top_centers,
    }


def build_normalized_actual_path(smoothed_points, barrel_geometry):
    if not smoothed_points or not barrel_geometry:
        return {"normalized_path": [], "transform": None}

    top = barrel_geometry.get("top")
    lower_left = barrel_geometry.get("lower_left")
    lower_right = barrel_geometry.get("lower_right")

    if top is None or lower_left is None or lower_right is None:
        return {"normalized_path": [], "transform": None}

    left_x = float(lower_left["center_x"])
    right_x = float(lower_right["center_x"])
    top_y = float(top["center_y"])

    if abs(right_x - left_x) < 1e-6:
        return {"normalized_path": [], "transform": None}

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

        normalized.append(
            (
                clamp(float(nx), 0.02, 0.98),
                clamp(float(ny), 0.02, 0.98),
            )
        )

    return {
        "normalized_path": normalized,
        "transform": {
            "left_x": round(left_x, 3),
            "right_x": round(right_x, 3),
            "top_y": round(top_y, 3),
            "max_path_y": round(max_path_y, 3),
        },
    }


def resample_polyline(points, num_samples=100):
    if not points:
        return []
    if len(points) == 1:
        return [points[0] for _ in range(num_samples)]

    points = dedupe_points(points, min_dist=1.0)
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
        (0.50, 0.94),
        (0.46, 0.90),
        (0.40, 0.86),
        (0.34, 0.82),
        (0.28, 0.77),
        (0.22, 0.72),
        (0.18, 0.66),
        (0.18, 0.60),
        (0.22, 0.56),
        (0.30, 0.56),
        (0.38, 0.60),
        (0.46, 0.65),
        (0.56, 0.68),
        (0.66, 0.68),
        (0.74, 0.66),
        (0.80, 0.62),
        (0.82, 0.58),
        (0.80, 0.54),
        (0.74, 0.52),
        (0.66, 0.54),
        (0.58, 0.58),
        (0.54, 0.64),
        (0.52, 0.56),
        (0.51, 0.46),
        (0.50, 0.36),
        (0.49, 0.28),
        (0.48, 0.22),
        (0.46, 0.16),
        (0.44, 0.12),
        (0.44, 0.10),
        (0.46, 0.08),
        (0.50, 0.08),
        (0.54, 0.08),
        (0.56, 0.10),
        (0.56, 0.13),
        (0.54, 0.18),
        (0.52, 0.24),
        (0.51, 0.32),
        (0.50, 0.42),
        (0.50, 0.54),
        (0.48, 0.66),
        (0.46, 0.78),
        (0.45, 0.88),
        (0.45, 0.94),
    ]


def build_ideal_template_path(direction, num_samples=100):
    left_points = build_left_first_ideal_waypoints()
    path = mirror_points_horiz(left_points) if direction == "right" else left_points
    return resample_polyline(path, num_samples=num_samples)


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
    smoothed_points,
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

    for i in range(1, len(smoothed_points)):
        p1 = (
            int(smoothed_points[i - 1][0] * scale_x),
            int(smoothed_points[i - 1][1] * scale_y),
        )
        p2 = (
            int(smoothed_points[i][0] * scale_x),
            int(smoothed_points[i][1] * scale_y),
        )
        cv2.line(canvas, p1, p2, (255, 0, 0), 3)

    for pt in smoothed_points:
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
        mini_w = 240
        mini_h = 300
        origin_x = max(20, canvas_width - mini_w - 20)
        origin_y = 20

        cv2.rectangle(
            canvas,
            (origin_x, origin_y),
            (origin_x + mini_w, origin_y + mini_h),
            (235, 235, 235),
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

        draw_norm_polyline(template_path, (170, 170, 170), 2)
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

                best_candidate = choose_best_horse_candidate(
                    candidates,
                    previous_accepted_point,
                    original_width,
                    original_height,
                )

                if idx % 2 == 0:
                    barrel_detections_resized = detect_barrels_in_frame(
                        inference_frame,
                        barrel_model,
                        confidence_threshold=BARREL_CONFIDENCE_THRESHOLD,
                    )
                    barrel_detections = scale_barrels_to_original(
                        barrel_detections_resized,
                        x_ratio,
                        y_ratio,
                    )

                all_barrel_detections.append(
                    {
                        "frame_index": int(target_frame),
                        "timestamp_seconds": timestamp_seconds,
                        "barrels": barrel_detections,
                    }
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
                            if jump_distance <= max_jump * 1.25:
                                accepted_points.append(current_point)
                                chosen_point_for_track = current_point
                                previous_accepted_point = current_point
                            else:
                                rejected_jump_count += 1
                                rejection_reason = f"jump_rejected_{round(jump_distance, 2)}px"
                else:
                    missed_detection_count += 1

                frame_file = os.path.join(output_dir, f"frame_{idx:03d}.jpg")
                if safe_imwrite(frame_file, frame) and os.path.exists(frame_file):
                    image_path = frame_file

                overlay = draw_overlay(
                    frame,
                    horse_detection,
                    barrel_detections,
                    None,
                    int(target_frame),
                    timestamp_seconds,
                )
                overlay_file = os.path.join(output_dir, f"frame_{idx:03d}_overlay.jpg")
                if safe_imwrite(overlay_file, overlay) and os.path.exists(overlay_file):
                    overlay_image_path = overlay_file

            track_points.append(chosen_point_for_track)

            sampled_frames.append(
                {
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
                }
            )

        interpolated_track = interpolate_small_gaps(track_points, max_gap=INTERPOLATION_MAX_GAP)
        smoothed_points = [p for p in interpolated_track if p is not None]
        smoothed_points = smooth_points(smoothed_points, alpha=SMOOTHING_ALPHA)
        smoothed_points = dedupe_points(smoothed_points, min_dist=6.0)

        barrel_detection_summary = summarize_barrel_detections(all_barrel_detections)

        provisional_identified_barrels, barrel_geometry = identify_barrels_simple(all_barrel_detections)
        provisional_frame_metrics = build_frame_metrics(sampled_frames, provisional_identified_barrels)

        pattern_direction_info = detect_pattern_direction(provisional_frame_metrics)
        actual_to_provisional_map = pattern_direction_info["actual_to_provisional_map"]

        identified_barrels = remap_barrel_keyed_dict(
            provisional_identified_barrels,
            actual_to_provisional_map,
        )
        frame_metrics = remap_frame_metric_labels(
            provisional_frame_metrics,
            actual_to_provisional_map,
        )

        turns = build_turns(frame_metrics)
        turns = enforce_turn_order(turns, pattern_direction_info["pattern_direction"])
        splits = build_splits(turns, frame_metrics)
        highlights = build_highlights(frame_metrics)

        direction = pattern_direction_info.get("pattern_direction") or "left"
        ideal_template_path = build_ideal_template_path(direction, num_samples=100)

        normalized_actual_path_result = build_normalized_actual_path(smoothed_points, barrel_geometry)
        normalized_actual_path = normalized_actual_path_result["normalized_path"]

        path_map_path = None
        if len(smoothed_points) > 2:
            path_map_path = f"{video_path}_path_map.jpg"
            save_path_map(
                original_width,
                original_height,
                smoothed_points,
                path_map_path,
                identified_barrels=identified_barrels,
                normalized_actual_path=[round_point(p, 4) for p in normalized_actual_path],
                template_path=[round_point(p, 4) for p in ideal_template_path],
            )

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

        insights = []

        if horse_detected_count < max(5, len(sample_indices) // 2):
            insights.append("Horse detection was somewhat limited, so path confidence is reduced.")

        if barrel_detection_summary["detected_frame_count"] < 2:
            insights.append("Barrel detections were limited, so barrel placement confidence is reduced.")

        if pattern_direction_info["method"] == "fallback_left":
            insights.append("Pattern direction was estimated conservatively because the first approach was not strongly confirmed.")

        if highlights["focus_next"]:
            insights.append(highlights["focus_next"])

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
            "barrel_metrics": {
                "barrel1": None,
                "barrel2": None,
                "barrel3": None,
            },
            "normalized_actual_path_transform": normalized_actual_path_result["transform"],
            "normalized_actual_template_path": [round_point(p, 4) for p in normalized_actual_path],
            "ideal_template_path": [round_point(p, 4) for p in ideal_template_path],
            "template_path_comparison": None,
            "speed_scores": {
                "barrel1": None,
                "barrel2": None,
                "barrel3": None,
            },
            "highlights": highlights,
            "frame_metrics": frame_metrics,
            "motion_samples": [],
            "sampled_frames": sampled_frames,
            "insights": insights[:4],
        }

        emit_json(output)

    except Exception as runtime_error:
        log_err("Python analysis crashed:", str(runtime_error))
        log_err(traceback.format_exc())

        emit_json(
            {
                "ok": False,
                "error": "Python analysis crashed",
                "details": str(runtime_error),
                "traceback": traceback.format_exc(),
                "video_path": video_path,
            }
        )
    finally:
        cap.release()


if __name__ == "__main__":
    main()