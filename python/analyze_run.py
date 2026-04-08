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

# Detection thresholds
HORSE_CONFIDENCE_THRESHOLD = 0.18
BARREL_CONFIDENCE_THRESHOLD = 0.48

# Sampling
TARGET_SAMPLE_FPS = 2.0
MIN_SAMPLED_FRAMES = 12
MAX_SAMPLED_FRAMES = 20

# Inference resolution
MAX_INFERENCE_WIDTH = 576
MAX_INFERENCE_HEIGHT = 324

# Smoothing
SMOOTHING_ALPHA = 0.34
INTERPOLATION_MAX_GAP = 3

# Canonical barrel positions in normalized space
CANONICAL_LEFT_BARREL_X = 0.24
CANONICAL_RIGHT_BARREL_X = 0.76
CANONICAL_TOP_BARREL_X = 0.50
CANONICAL_TOP_BARREL_Y = 0.22
CANONICAL_LOWER_BARREL_Y = 0.68
CANONICAL_HOME_Y = 0.94

# How close (fraction of arena diagonal) the horse must be to count as a turn detection
BARREL_NEAR_RADIUS_FRACTION = 0.28


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
        indices = sorted(set(int(round(i * (frame_count - 1) / max(desired - 1, 1))) for i in range(desired)))
    if len(indices) > MAX_SAMPLED_FRAMES:
        desired = MAX_SAMPLED_FRAMES
        indices = sorted(set(int(round(i * (frame_count - 1) / max(desired - 1, 1))) for i in range(desired)))
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
        x1 *= x_ratio; y1 *= y_ratio; x2 *= x_ratio; y2 *= y_ratio
        cx, cy = compute_bottom_center((x1, y1, x2, y2))
        area = max(0.0, (x2 - x1) * (y2 - y1))
        h = max(0.0, y2 - y1)
        tracking_point = (float(cx), float(y2 - h * 0.10))
        candidates.append({
            "confidence": round(float(conf), 4),
            "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
            "center": [round(float(cx), 2), round(float(cy), 2)],
            "tracking_point": tracking_point,
            "area": float(area),
        })
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
        results = barrel_model.predict(source=frame, conf=confidence_threshold, verbose=False)
    barrels = []
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return barrels
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0].item())
        barrels.append({
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "confidence": round(conf, 3),
            "center_x": float((x1 + x2) / 2.0),
            "center_y": float((y1 + y2) / 2.0),
        })
    return barrels


def scale_barrels_to_original(barrels, x_ratio, y_ratio):
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


def adaptive_max_jump(width, height):
    diagonal = math.hypot(width, height) if width > 0 and height > 0 else 1000.0
    return clamp(diagonal * 0.15, 100.0, 320.0)


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
        if (gap_start >= 0 and gap_end < n and
                result[gap_start] is not None and result[gap_end] is not None and
                gap_len <= max_gap):
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


def average_values(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def identify_barrels_simple(all_barrel_detections):
    points = []
    for frame_entry in all_barrel_detections:
        for barrel in frame_entry["barrels"]:
            points.append({
                "x": float(barrel["center_x"]),
                "y": float(barrel["center_y"]),
                "confidence": float(barrel["confidence"]),
            })

    result = {"barrel1": None, "barrel2": None, "barrel3": None}
    geometry = {"top": None, "lower_left": None, "lower_right": None}

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
        "center_x": top_x, "center_y": top_y,
        "detection_count": len(top_candidates),
        "average_confidence": average_values([p["confidence"] for p in top_candidates]),
    }

    geometry["top"] = {"center_x": round(top_cluster["center_x"], 2), "center_y": round(top_cluster["center_y"], 2), "detection_count": int(top_cluster["detection_count"]), "average_confidence": round(top_cluster["average_confidence"], 3)}
    geometry["lower_left"] = {"center_x": round(lower_left["center_x"], 2), "center_y": round(lower_left["center_y"], 2), "detection_count": int(lower_left["detection_count"]), "average_confidence": round(lower_left["average_confidence"], 3)}
    geometry["lower_right"] = {"center_x": round(lower_right["center_x"], 2), "center_y": round(lower_right["center_y"], 2), "detection_count": int(lower_right["detection_count"]), "average_confidence": round(lower_right["average_confidence"], 3)}

    result["barrel1"] = {"center_x": round(lower_left["center_x"], 2), "center_y": round(lower_left["center_y"], 2), "detection_count": int(lower_left["detection_count"]), "average_confidence": round(lower_left["average_confidence"], 3), "geometry_role": "lower_left"}
    result["barrel2"] = {"center_x": round(lower_right["center_x"], 2), "center_y": round(lower_right["center_y"], 2), "detection_count": int(lower_right["detection_count"]), "average_confidence": round(lower_right["average_confidence"], 3), "geometry_role": "lower_right"}
    result["barrel3"] = {"center_x": round(top_cluster["center_x"], 2), "center_y": round(top_cluster["center_y"], 2), "detection_count": int(top_cluster["detection_count"]), "average_confidence": round(top_cluster["average_confidence"], 3), "geometry_role": "top"}

    return result, geometry


def build_frame_metrics(sampled_frames, identified_barrels):
    metrics = []
    for frame in sampled_frames:
        horse_detection = frame.get("horse_detection")
        horse_center = None
        nearest_barrel = None
        nearest_distance = None
        dist_map = {"barrel1": None, "barrel2": None, "barrel3": None}

        if horse_detection is not None:
            horse_center = (float(horse_detection["tracking_point"][0]), float(horse_detection["tracking_point"][1]))
            for barrel_name in ("barrel1", "barrel2", "barrel3"):
                barrel_info = identified_barrels.get(barrel_name)
                if barrel_info is None:
                    continue
                barrel_center = (float(barrel_info["center_x"]), float(barrel_info["center_y"]))
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
            "actual_to_provisional_map": {"barrel1": "barrel1", "barrel2": "barrel2", "barrel3": "barrel3"},
            "reason": "defaulted to left-first",
            "confidence": 0.5,
            "method": "fallback_left",
        }

    vote_window = lower_frames[: min(4, len(lower_frames))]
    left_votes = sum(1 for m in vote_window if m["nearest_barrel"] == "barrel1")
    right_votes = sum(1 for m in vote_window if m["nearest_barrel"] == "barrel2")

    if right_votes > left_votes:
        return {
            "pattern_direction": "right",
            "actual_to_provisional_map": {"barrel1": "barrel2", "barrel2": "barrel1", "barrel3": "barrel3"},
            "reason": "early lower-barrel approach favored the right side",
            "confidence": 0.78,
            "method": "early_lower_votes",
        }

    return {
        "pattern_direction": "left",
        "actual_to_provisional_map": {"barrel1": "barrel1", "barrel2": "barrel2", "barrel3": "barrel3"},
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
        new_metric["dist_to_barrel1_px"] = metric.get(f"dist_to_{actual_to_provisional_map['barrel1']}_px")
        new_metric["dist_to_barrel2_px"] = metric.get(f"dist_to_{actual_to_provisional_map['barrel2']}_px")
        new_metric["dist_to_barrel3_px"] = metric.get(f"dist_to_{actual_to_provisional_map['barrel3']}_px")
        provisional_nearest = metric.get("nearest_barrel")
        new_metric["nearest_barrel"] = (provisional_to_actual.get(provisional_nearest) if provisional_nearest is not None else None)
        remapped.append(new_metric)
    return remapped


def find_barrel_apex(barrel_name, frame_metrics, min_approach_frames=2):
    dist_key = f"dist_to_{barrel_name}_px"
    valid = [m for m in frame_metrics if m["horse_detected"] and m.get(dist_key) is not None]
    if len(valid) < min_approach_frames:
        return None
    min_metric = min(valid, key=lambda m: m[dist_key])
    min_idx = valid.index(min_metric)
    if min_idx == 0:
        if len(valid) > 1:
            remaining = valid[1:]
            min_metric = min(remaining, key=lambda m: m[dist_key])
        else:
            return None
    return {
        "barrel_name": barrel_name,
        "start_frame": int(min_metric["frame_index"]),
        "apex_frame": int(min_metric["frame_index"]),
        "end_frame": int(min_metric["frame_index"]),
        "start_timestamp_seconds": min_metric["timestamp_seconds"],
        "apex_timestamp_seconds": min_metric["timestamp_seconds"],
        "end_timestamp_seconds": min_metric["timestamp_seconds"],
        "min_distance_px": round_or_none(min_metric[dist_key], 2),
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
        return {"start_to_barrel1_seconds": None, "barrel1_to_barrel2_seconds": None, "barrel2_to_barrel3_seconds": None, "barrel3_to_home_seconds": None}

    b1 = turns.get("barrel1")
    b2 = turns.get("barrel2")
    b3 = turns.get("barrel3")
    start_time = valid[0]["timestamp_seconds"]
    end_time = valid[-1]["timestamp_seconds"]

    s1 = None
    if b1 and b1["apex_timestamp_seconds"] is not None:
        raw_s1 = b1["apex_timestamp_seconds"] - start_time
        if raw_s1 >= 0.5:
            s1 = raw_s1

    s2 = (b2["apex_timestamp_seconds"] - b1["apex_timestamp_seconds"] if b1 and b2 and b1["apex_timestamp_seconds"] is not None and b2["apex_timestamp_seconds"] is not None else None)
    s3 = (b3["apex_timestamp_seconds"] - b2["apex_timestamp_seconds"] if b2 and b3 and b2["apex_timestamp_seconds"] is not None and b3["apex_timestamp_seconds"] is not None else None)
    s4 = (end_time - b3["apex_timestamp_seconds"] if b3 and b3["apex_timestamp_seconds"] is not None else None)

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
    barrel_distances = {"barrel1": [], "barrel2": [], "barrel3": []}
    for metric in frame_metrics:
        for barrel_name in ("barrel1", "barrel2", "barrel3"):
            d = metric.get(f"dist_to_{barrel_name}_px")
            if d is not None:
                barrel_distances[barrel_name].append(float(d))

    avg_distances = {k: (sum(v) / len(v) if v else None) for k, v in barrel_distances.items()}
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

    return {"best_barrel": best_barrel, "best_turn": best_turn, "focus_next": focus_next}


def summarize_barrel_detections(all_barrel_detections):
    centers = []
    for frame_entry in all_barrel_detections:
        for barrel in frame_entry["barrels"]:
            centers.append({"center_x": barrel["center_x"], "center_y": barrel["center_y"], "confidence": barrel["confidence"], "frame_index": frame_entry["frame_index"]})

    if not centers:
        return {"detected_frame_count": 0, "total_barrel_boxes": 0, "average_barrels_per_detected_frame": 0.0, "top_barrel_centers": []}

    sorted_centers = sorted(centers, key=lambda c: c["confidence"], reverse=True)
    top_centers = [{"center_x": c["center_x"], "center_y": c["center_y"], "confidence": c["confidence"], "frame_index": c["frame_index"]} for c in sorted_centers[:8]]
    detected_frame_count = sum(1 for entry in all_barrel_detections if len(entry["barrels"]) > 0)
    total_barrel_boxes = len(centers)
    avg = total_barrel_boxes / detected_frame_count if detected_frame_count > 0 else 0.0
    return {"detected_frame_count": detected_frame_count, "total_barrel_boxes": total_barrel_boxes, "average_barrels_per_detected_frame": round(avg, 3), "top_barrel_centers": top_centers}


# ─── Ideal Path ───────────────────────────────────────────────────────────────

def build_left_first_ideal_waypoints():
    """
    Hardcoded ideal left-first barrel pattern waypoints in normalized 0-1 space.
    This is the same path shown as the teal line in the frontend.
    """
    return [
        (0.42, 0.95), (0.42, 0.88), (0.42, 0.80), (0.42, 0.72), (0.42, 0.65),
        (0.48, 0.62), (0.56, 0.59), (0.64, 0.57), (0.72, 0.56), (0.80, 0.57),
        (0.86, 0.59), (0.90, 0.55), (0.90, 0.50), (0.87, 0.46), (0.82, 0.44),
        (0.74, 0.44), (0.67, 0.47), (0.63, 0.52), (0.63, 0.57), (0.66, 0.61),
        (0.72, 0.64), (0.64, 0.65), (0.54, 0.66), (0.42, 0.66), (0.30, 0.65),
        (0.20, 0.63), (0.12, 0.60), (0.07, 0.56), (0.06, 0.51), (0.08, 0.46),
        (0.13, 0.43), (0.20, 0.42), (0.28, 0.44), (0.33, 0.49), (0.33, 0.55),
        (0.30, 0.60), (0.26, 0.64), (0.34, 0.65), (0.40, 0.64), (0.42, 0.60),
        (0.42, 0.52), (0.42, 0.43), (0.42, 0.34), (0.42, 0.26), (0.42, 0.20),
        (0.43, 0.15), (0.46, 0.11), (0.50, 0.09), (0.55, 0.10), (0.58, 0.14),
        (0.58, 0.19), (0.55, 0.23), (0.50, 0.25), (0.45, 0.23), (0.43, 0.19),
        (0.42, 0.25), (0.42, 0.34), (0.42, 0.44), (0.42, 0.54), (0.42, 0.64),
        (0.42, 0.74), (0.42, 0.84), (0.42, 0.95),
    ]


def mirror_points_horiz(points):
    return [(1.0 - float(x), float(y)) for x, y in points]


def resample_polyline(points, num_samples=100):
    if not points:
        return []
    if len(points) == 1:
        return [points[0] for _ in range(num_samples)]
    points = list(dedupe_points(list(points), min_dist=0.001))
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


def build_ideal_template_path(direction, num_samples=100):
    left_points = build_left_first_ideal_waypoints()
    path = mirror_points_horiz(left_points) if direction == "right" else left_points
    return resample_polyline(path, num_samples=num_samples)


# ─── Warped Actual Path ───────────────────────────────────────────────────────

def get_closest_horse_point_to_barrel(frame_metrics, barrel_name, barrel_center_px, arena_diagonal):
    """
    Find the actual horse tracking point closest to a barrel.
    This is the most reliable single data point we have for each turn.
    Returns the horse position in pixel space, or None if not found.
    """
    dist_key = f"dist_to_{barrel_name}_px"
    radius = arena_diagonal * BARREL_NEAR_RADIUS_FRACTION

    candidates = []
    for metric in frame_metrics:
        if not metric["horse_detected"] or metric["horse_center"] is None:
            continue
        dist = metric.get(dist_key)
        if dist is not None and dist <= radius:
            candidates.append({
                "point": (float(metric["horse_center"][0]), float(metric["horse_center"][1])),
                "dist": dist,
            })

    if not candidates:
        return None

    # Return the point closest to the barrel
    best = min(candidates, key=lambda c: c["dist"])
    return best["point"]


def build_warped_actual_path(
    frame_metrics,
    identified_barrels,
    barrel_geometry,
    direction,
    original_width,
    original_height,
):
    """
    Build the "Your Path" line by:
    1. Starting with the hardcoded ideal path (always looks like a barrel run)
    2. At each barrel, measuring where the horse actually was vs where the ideal path passes
    3. Warping the ideal path toward the actual horse position at each barrel zone
    4. Smoothly blending between warped and unwarped sections

    Result: always looks like a barrel pattern, but shows real deviation at each turn.
    """

    if not identified_barrels or not barrel_geometry:
        return []

    top = barrel_geometry.get("top")
    lower_left = barrel_geometry.get("lower_left")
    lower_right = barrel_geometry.get("lower_right")

    if top is None or lower_left is None or lower_right is None:
        return []

    arena_diagonal = math.hypot(original_width, original_height)

    # Get detected barrel centers in pixel space
    b1 = identified_barrels.get("barrel1")
    b2 = identified_barrels.get("barrel2")
    b3 = identified_barrels.get("barrel3")

    if not b1 or not b2 or not b3:
        return []

    b1_px = (float(b1["center_x"]), float(b1["center_y"]))
    b2_px = (float(b2["center_x"]), float(b2["center_y"]))
    b3_px = (float(b3["center_x"]), float(b3["center_y"]))

    # Find actual horse positions near each barrel in pixel space
    h1_px = get_closest_horse_point_to_barrel(frame_metrics, "barrel1", b1_px, arena_diagonal)
    h2_px = get_closest_horse_point_to_barrel(frame_metrics, "barrel2", b2_px, arena_diagonal)
    h3_px = get_closest_horse_point_to_barrel(frame_metrics, "barrel3", b3_px, arena_diagonal)

    # Build coordinate transform from pixel space to normalized space
    left_x = float(lower_left["center_x"])
    right_x = float(lower_right["center_x"])
    top_y = float(top["center_y"])

    if abs(right_x - left_x) < 1.0:
        return []

    # Estimate home y in pixel space
    home_y_px = top_y + (original_height - top_y) * 0.85

    def px_to_norm(px_point):
        """Convert pixel coordinates to normalized 0-1 space."""
        x, y = px_point
        nx = CANONICAL_LEFT_BARREL_X + (
            (float(x) - left_x) / (right_x - left_x)
        ) * (CANONICAL_RIGHT_BARREL_X - CANONICAL_LEFT_BARREL_X)
        ny = CANONICAL_TOP_BARREL_Y + (
            (float(y) - top_y) / max(home_y_px - top_y, 1e-9)
        ) * (CANONICAL_HOME_Y - CANONICAL_TOP_BARREL_Y)
        return (clamp(float(nx), 0.02, 0.98), clamp(float(ny), 0.02, 0.98))

    # Get the ideal path in normalized space (same as teal line)
    ideal_path = build_ideal_template_path(direction, num_samples=80)

    # Convert actual horse barrel positions to normalized space
    h1_norm = px_to_norm(h1_px) if h1_px else None
    h2_norm = px_to_norm(h2_px) if h2_px else None
    h3_norm = px_to_norm(h3_px) if h3_px else None

    # Ideal barrel positions in normalized space (where the ideal path passes near each barrel)
    # These are the canonical positions the ideal path is built around
    if direction == "right":
        ideal_b1_norm = (0.78, 0.50)  # right barrel
        ideal_b2_norm = (0.22, 0.50)  # left barrel
        ideal_b3_norm = (0.50, 0.17)  # top barrel
    else:
        ideal_b1_norm = (0.22, 0.50)  # left barrel
        ideal_b2_norm = (0.78, 0.50)  # right barrel
        ideal_b3_norm = (0.50, 0.17)  # top barrel

    # Calculate offsets: how far did the horse deviate from the ideal at each barrel
    # Positive offset means horse was wider/different than ideal
    offsets = {}

    if h1_norm:
        offsets["barrel1"] = (
            h1_norm[0] - ideal_b1_norm[0],
            h1_norm[1] - ideal_b1_norm[1],
        )
    else:
        offsets["barrel1"] = (0.0, 0.0)

    if h2_norm:
        offsets["barrel2"] = (
            h2_norm[0] - ideal_b2_norm[0],
            h2_norm[1] - ideal_b2_norm[1],
        )
    else:
        offsets["barrel2"] = (0.0, 0.0)

    if h3_norm:
        offsets["barrel3"] = (
            h3_norm[0] - ideal_b3_norm[0],
            h3_norm[1] - ideal_b3_norm[1],
        )
    else:
        offsets["barrel3"] = (0.0, 0.0)

    # Now warp the ideal path using these offsets
    # The warp influence falls off with distance from each barrel
    # so the path only deviates near each barrel, not everywhere
    WARP_RADIUS = 0.25  # normalized radius of influence around each barrel

    def warp_influence(point, barrel_norm, radius):
        """Gaussian falloff — full warp at barrel, zero warp far away."""
        d = distance(point, barrel_norm)
        if d >= radius:
            return 0.0
        return 1.0 - (d / radius)

    warped_path = []
    for pt in ideal_path:
        x, y = float(pt[0]), float(pt[1])

        # Calculate total warp at this point from all three barrels
        total_offset_x = 0.0
        total_offset_y = 0.0
        total_weight = 0.0

        for barrel_name, barrel_norm in [
            ("barrel1", ideal_b1_norm),
            ("barrel2", ideal_b2_norm),
            ("barrel3", ideal_b3_norm),
        ]:
            influence = warp_influence((x, y), barrel_norm, WARP_RADIUS)
            if influence > 0:
                ox, oy = offsets[barrel_name]
                total_offset_x += influence * ox
                total_offset_y += influence * oy
                total_weight += influence

        if total_weight > 0:
            # Cap the maximum warp to prevent extreme distortion
            max_warp = 0.20
            total_offset_x = clamp(total_offset_x, -max_warp, max_warp)
            total_offset_y = clamp(total_offset_y, -max_warp, max_warp)
            x += total_offset_x
            y += total_offset_y

        warped_path.append([
            round(clamp(x, 0.02, 0.98), 4),
            round(clamp(y, 0.02, 0.98), 4),
        ])

    return warped_path


# ─── Draw / Save ──────────────────────────────────────────────────────────────

def draw_overlay(frame, horse_detection, barrel_detections, identified_barrels, frame_index, timestamp_seconds):
    overlay = frame.copy()
    for barrel in barrel_detections or []:
        x1 = int(barrel["x1"]); y1 = int(barrel["y1"]); x2 = int(barrel["x2"]); y2 = int(barrel["y2"])
        conf = barrel.get("confidence")
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(overlay, f"barrel {conf}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2, cv2.LINE_AA)
    if horse_detection:
        x1, y1, x2, y2 = [int(v) for v in horse_detection["bbox"]]
        cx, cy = [int(v) for v in horse_detection["tracking_point"]]
        conf = horse_detection.get("confidence")
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(overlay, f"horse {conf}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"frame {frame_index}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if timestamp_seconds is not None:
        cv2.putText(overlay, f"time {timestamp_seconds}s", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def save_path_map(width, height, smoothed_points, out_path, identified_barrels=None):
    canvas_width = min(width, 960)
    canvas_height = min(height, 540)
    scale_x = canvas_width / float(max(width, 1))
    scale_y = canvas_height / float(max(height, 1))
    canvas = 255 * (cv2.UMat(canvas_height, canvas_width, cv2.CV_8UC3).get() * 0 + 1)
    for i in range(1, len(smoothed_points)):
        p1 = (int(smoothed_points[i - 1][0] * scale_x), int(smoothed_points[i - 1][1] * scale_y))
        p2 = (int(smoothed_points[i][0] * scale_x), int(smoothed_points[i][1] * scale_y))
        cv2.line(canvas, p1, p2, (255, 0, 0), 3)
    if identified_barrels:
        for barrel_name, barrel_info in identified_barrels.items():
            if barrel_info is None:
                continue
            x = int(barrel_info["center_x"] * scale_x)
            y = int(barrel_info["center_y"] * scale_y)
            cv2.circle(canvas, (x, y), 12, (0, 255, 255), 2)
            cv2.putText(canvas, barrel_name.upper(), (x + 8, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2, cv2.LINE_AA)
    safe_imwrite(out_path, canvas)


# ─── Main ─────────────────────────────────────────────────────────────────────

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
            fail("Video metadata could not be read correctly.", {"video_path": video_path, "frame_count": frame_count, "fps": fps, "width": original_width, "height": original_height})
            return

        log_err(f"Video opened. frame_count={frame_count}, fps={round(fps, 3)}, width={original_width}, height={original_height}, duration={round(duration, 3)}")

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
                    horse_results = horse_model.predict(source=inference_frame, conf=HORSE_CONFIDENCE_THRESHOLD, classes=[HORSE_CLASS_ID], verbose=False)

                candidates = []
                if horse_results and len(horse_results) > 0:
                    candidates = build_horse_candidates(horse_results[0], x_ratio=x_ratio, y_ratio=y_ratio)

                best_candidate = choose_best_horse_candidate(candidates, previous_accepted_point, original_width, original_height)

                if idx % 2 == 0:
                    barrel_detections_resized = detect_barrels_in_frame(inference_frame, barrel_model, confidence_threshold=BARREL_CONFIDENCE_THRESHOLD)
                    barrel_detections = scale_barrels_to_original(barrel_detections_resized, x_ratio, y_ratio)

                all_barrel_detections.append({"frame_index": int(target_frame), "timestamp_seconds": timestamp_seconds, "barrels": barrel_detections})

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
                        elif jump_distance <= max_jump * 1.10:
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

                overlay = draw_overlay(frame, horse_detection, barrel_detections, None, int(target_frame), timestamp_seconds)
                overlay_file = os.path.join(output_dir, f"frame_{idx:03d}_overlay.jpg")
                if safe_imwrite(overlay_file, overlay) and os.path.exists(overlay_file):
                    overlay_image_path = overlay_file

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

        # ── Process results ───────────────────────────────────────────────────
        interpolated_track = interpolate_small_gaps(track_points, max_gap=INTERPOLATION_MAX_GAP)
        smoothed_points = [p for p in interpolated_track if p is not None]
        smoothed_points = smooth_points(smoothed_points, alpha=SMOOTHING_ALPHA)
        smoothed_points = dedupe_points(smoothed_points, min_dist=6.0)

        barrel_detection_summary = summarize_barrel_detections(all_barrel_detections)
        provisional_identified_barrels, barrel_geometry = identify_barrels_simple(all_barrel_detections)
        provisional_frame_metrics = build_frame_metrics(sampled_frames, provisional_identified_barrels)
        pattern_direction_info = detect_pattern_direction(provisional_frame_metrics)
        actual_to_provisional_map = pattern_direction_info["actual_to_provisional_map"]

        identified_barrels = remap_barrel_keyed_dict(provisional_identified_barrels, actual_to_provisional_map)
        frame_metrics = remap_frame_metric_labels(provisional_frame_metrics, actual_to_provisional_map)

        turns = build_turns(frame_metrics)
        turns = enforce_turn_order(turns, pattern_direction_info["pattern_direction"])
        splits = build_splits(turns, frame_metrics)
        highlights = build_highlights(frame_metrics)

        direction = pattern_direction_info.get("pattern_direction") or "left"

        # ── Build ideal template path ─────────────────────────────────────────
        ideal_template_path = build_ideal_template_path(direction, num_samples=80)

        # ── Build warped actual path ──────────────────────────────────────────
        # This is the "Your Path" line — ideal path warped by actual horse
        # positions detected near each barrel. Always looks like a barrel run
        # but shows real deviation at each turn.
        warped_actual_path = build_warped_actual_path(
            frame_metrics,
            identified_barrels,
            barrel_geometry,
            direction,
            original_width,
            original_height,
        )

        # Fall back to ideal path if warp fails (e.g. barrels not detected)
        if not warped_actual_path:
            warped_actual_path = [[round(p[0], 4), round(p[1], 4)] for p in ideal_template_path]

        path_map_path = None
        if len(smoothed_points) > 2:
            path_map_path = f"{video_path}_path_map.jpg"
            save_path_map(original_width, original_height, smoothed_points, path_map_path, identified_barrels=identified_barrels)

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

        output = {
            "ok": True,
            "message": "Barrel path analysis completed.",
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
            # This is what the frontend renders as "Your Path"
            "normalized_smoothed_path_points": warped_actual_path,
            "path_map_path": path_map_path,
            "tracking_quality": tracking_quality,
            "barrel_detection_summary": barrel_detection_summary,
            "barrel_geometry": barrel_geometry,
            "pattern_direction": direction,
            "pattern_direction_info": pattern_direction_info,
            "identified_barrels": identified_barrels,
            "turns": turns,
            "splits": splits,
            "barrel_metrics": {"barrel1": None, "barrel2": None, "barrel3": None},
            "normalized_actual_path_transform": None,
            "normalized_actual_template_path": warped_actual_path,
            "ideal_template_path": [[round(p[0], 4), round(p[1], 4)] for p in ideal_template_path],
            "template_path_comparison": None,
            "speed_scores": {"barrel1": None, "barrel2": None, "barrel3": None},
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