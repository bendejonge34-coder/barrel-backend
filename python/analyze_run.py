import os
import sys
import json
import math
import cv2
import contextlib
import io
import traceback

# ─── Environment Setup ────────────────────────────────────────────────────────

os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["PYTHONUNBUFFERED"] = "1"

from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BARREL_MODEL_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "runs_detect",
    "barrel_detector_local", "weights", "best.pt"
)
HORSE_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# ─── Detection Constants ──────────────────────────────────────────────────────

HORSE_CLASS_ID = 17
HORSE_CONFIDENCE_THRESHOLD = 0.18
BARREL_CONFIDENCE_THRESHOLD = 0.45

# Sampling — reduced for minimal analysis (was 4.0 / 48 / 12.0 / 1.5)
TARGET_SAMPLE_FPS = 2.0        # Reduced: 2 frames per second is enough for splits
MIN_SAMPLED_FRAMES = 12        # Reduced from 16
MAX_SAMPLED_FRAMES = 24        # Reduced from 48

# Dense sampling around barrel apex zones
DENSE_SAMPLE_FPS = 6.0         # Reduced from 12.0
DENSE_WINDOW_SECONDS = 1.0     # Reduced from 1.5

# Inference size — reduced for faster processing (was 768 x 432)
MAX_INFERENCE_WIDTH = 480
MAX_INFERENCE_HEIGHT = 270

# Trajectory
SMOOTHING_ALPHA = 0.30
INTERPOLATION_MAX_GAP = 4
MAX_JUMP_FRACTION = 0.15

# Barrel geometry canonical coords (normalized 0-1)
CANONICAL_LEFT_BARREL_X = 0.24
CANONICAL_RIGHT_BARREL_X = 0.76
CANONICAL_TOP_BARREL_Y = 0.22
CANONICAL_HOME_Y = 0.94
BARREL_NEAR_RADIUS_FRACTION = 0.28

# Split sanity bounds
MIN_SPLIT_SECONDS = 0.3
MAX_SPLIT_SECONDS = 15.0
MIN_RUN_TIME = 5.0
MAX_RUN_TIME = 60.0


# ─── Logging ──────────────────────────────────────────────────────────────────

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


# ─── Math Utilities ───────────────────────────────────────────────────────────

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def distance(p1, p2):
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))

def average_values(values):
    values = [float(v) for v in values if v is not None]
    return (sum(values) / len(values)) if values else None

def round_or_none(value, decimals=2):
    if value is None:
        return None
    return round(float(value), decimals)

def round_point(point, decimals=2):
    if point is None:
        return None
    return [round(float(point[0]), decimals), round(float(point[1]), decimals)]


# ─── Speed Calculation ────────────────────────────────────────────────────────

def calculate_speed_profile(frame_metrics, fps):
    """
    Calculate horse speed (pixels/second) at each frame.
    Used for ft/s display in split timing.
    """
    enriched = [dict(m) for m in frame_metrics]
    detected = [(i, m) for i, m in enumerate(enriched) if m.get("horse_center")]

    for idx, (i, m) in enumerate(detected):
        if idx == 0 or idx == len(detected) - 1:
            enriched[i]["speed_px_per_sec"] = None
            continue
        prev_i, prev_m = detected[idx - 1]
        next_i, next_m = detected[idx + 1]
        prev_center = prev_m.get("horse_center")
        next_center = next_m.get("horse_center")
        if not prev_center or not next_center:
            enriched[i]["speed_px_per_sec"] = None
            continue
        prev_ts = prev_m.get("timestamp_seconds")
        next_ts = next_m.get("timestamp_seconds")
        if prev_ts is None or next_ts is None or next_ts <= prev_ts:
            enriched[i]["speed_px_per_sec"] = None
            continue
        dist = distance(prev_center, next_center)
        time_diff = next_ts - prev_ts
        speed = dist / time_diff if time_diff > 0 else 0
        enriched[i]["speed_px_per_sec"] = round(speed, 2)

    for i, m in enumerate(enriched):
        if "speed_px_per_sec" not in m:
            enriched[i]["speed_px_per_sec"] = None

    return enriched


# ─── Speed Summary ────────────────────────────────────────────────────────────

def build_speed_summary(frame_metrics, turns, fps):
    """
    Build speed summary for split timing ft/s display.
    """
    speeds = [(m["timestamp_seconds"], m["speed_px_per_sec"])
              for m in frame_metrics
              if m.get("speed_px_per_sec") is not None and m.get("timestamp_seconds") is not None]

    if not speeds:
        return None

    all_speeds = [s[1] for s in speeds]
    avg_speed = average_values(all_speeds)
    max_speed = max(all_speeds)
    min_speed = min(all_speeds)

    b1_ts = turns.get("barrel1", {}) and turns["barrel1"].get("apex_timestamp_seconds") if turns.get("barrel1") else None
    b2_ts = turns.get("barrel2", {}) and turns["barrel2"].get("apex_timestamp_seconds") if turns.get("barrel2") else None
    b3_ts = turns.get("barrel3", {}) and turns["barrel3"].get("apex_timestamp_seconds") if turns.get("barrel3") else None

    def avg_speed_in_range(t_start, t_end):
        in_range = [s for ts, s in speeds if t_start is not None and t_end is not None and t_start <= ts <= t_end]
        return average_values(in_range)

    all_ts = [ts for ts, s in speeds]
    run_start = min(all_ts)
    run_end = max(all_ts)

    section_speeds = {
        "alley_to_barrel1": avg_speed_in_range(run_start, b1_ts),
        "barrel1_to_barrel2": avg_speed_in_range(b1_ts, b2_ts),
        "barrel2_to_barrel3": avg_speed_in_range(b2_ts, b3_ts),
        "barrel3_to_home": avg_speed_in_range(b3_ts, run_end),
    }

    valid_sections = {k: v for k, v in section_speeds.items() if v is not None}
    slowest_section = min(valid_sections, key=valid_sections.get) if valid_sections else None
    fastest_section = max(valid_sections, key=valid_sections.get) if valid_sections else None

    section_labels = {
        "alley_to_barrel1": "alley to first barrel",
        "barrel1_to_barrel2": "first to second barrel",
        "barrel2_to_barrel3": "second to third barrel",
        "barrel3_to_home": "third barrel to home",
    }

    return {
        "average_speed_px_per_sec": round(avg_speed, 1) if avg_speed else None,
        "max_speed_px_per_sec": round(max_speed, 1),
        "min_speed_px_per_sec": round(min_speed, 1),
        "section_speeds": {k: round(v, 1) if v else None for k, v in section_speeds.items()},
        "slowest_section": slowest_section,
        "slowest_section_label": section_labels.get(slowest_section),
        "fastest_section": fastest_section,
        "fastest_section_label": section_labels.get(fastest_section),
    }


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return YOLO(model_path)


# ─── Frame Utilities ──────────────────────────────────────────────────────────

def resize_for_inference(frame):
    if frame is None:
        return None, 1.0, 1.0
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame, 1.0, 1.0
    scale = min(MAX_INFERENCE_WIDTH / float(w), MAX_INFERENCE_HEIGHT / float(h), 1.0)
    if scale >= 1.0:
        return frame, 1.0, 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, w / float(new_w), h / float(new_h)

def read_frame_at(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    return frame if (ret and frame is not None) else None


# ─── Frame Index Builders ─────────────────────────────────────────────────────

def build_base_sample_indices(frame_count, fps):
    if frame_count <= 0 or fps <= 0:
        return []
    step = max(1, int(round(fps / TARGET_SAMPLE_FPS)))
    indices = list(range(0, frame_count, step))
    if not indices:
        indices = [0]
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    if len(indices) < MIN_SAMPLED_FRAMES and frame_count > MIN_SAMPLED_FRAMES:
        desired = min(MIN_SAMPLED_FRAMES, frame_count)
        indices = [int(round(i * (frame_count - 1) / max(desired - 1, 1))) for i in range(desired)]
    if len(indices) > MAX_SAMPLED_FRAMES:
        desired = MAX_SAMPLED_FRAMES
        indices = [int(round(i * (frame_count - 1) / max(desired - 1, 1))) for i in range(desired)]
    return sorted(set(indices))

def build_dense_indices_around_apexes(apex_timestamps, fps, frame_count, existing_indices):
    if not apex_timestamps or fps <= 0:
        return existing_indices
    dense_step = max(1, int(round(fps / DENSE_SAMPLE_FPS)))
    window_frames = int(DENSE_WINDOW_SECONDS * fps)
    extra = set()
    for ts in apex_timestamps:
        if ts is None:
            continue
        center_frame = int(ts * fps)
        start = max(0, center_frame - window_frames)
        end = min(frame_count - 1, center_frame + window_frames)
        for f in range(start, end + 1, dense_step):
            extra.add(f)
    return sorted(set(existing_indices) | extra)


# ─── Horse Detection ──────────────────────────────────────────────────────────

def detect_horse_candidates(result, x_ratio=1.0, y_ratio=1.0):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    candidates = []
    for box, cls_id, conf in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
    ):
        if int(cls_id) != HORSE_CLASS_ID:
            continue
        x1 = box[0] * x_ratio
        y1 = box[1] * y_ratio
        x2 = box[2] * x_ratio
        y2 = box[3] * y_ratio
        cx = (x1 + x2) / 2.0
        h = max(0.0, y2 - y1)
        area = max(0.0, (x2 - x1) * h)
        tracking_point = (float(cx), float(y2 - h * 0.10))
        candidates.append({
            "confidence": round(float(conf), 4),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "center": [round(cx, 2), round(float((y1 + y2) / 2.0), 2)],
            "tracking_point": tracking_point,
            "area": float(area),
        })
    return candidates

def choose_best_horse(candidates, prev_point, frame_width, frame_height):
    if not candidates:
        return None
    if prev_point is None:
        return max(candidates, key=lambda c: (c["confidence"] * 2.0 + c["area"] / 50000.0))
    diagonal = math.hypot(frame_width, frame_height) or 1.0
    best, best_score = None, None
    for c in candidates:
        norm_dist = distance(prev_point, c["tracking_point"]) / diagonal
        score = (c["confidence"] * 2.5) + min(c["area"] / 40000.0, 1.2) - (norm_dist * 2.4)
        if best_score is None or score > best_score:
            best_score = score
            best = c
    return best


# ─── Barrel Detection ─────────────────────────────────────────────────────────

def detect_barrels(frame, barrel_model, x_ratio=1.0, y_ratio=1.0):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        results = barrel_model.predict(source=frame, conf=BARREL_CONFIDENCE_THRESHOLD, verbose=False)
    barrels = []
    if not results or results[0].boxes is None:
        return barrels
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0].item())
        barrels.append({
            "x1": round(x1 * x_ratio, 2), "y1": round(y1 * y_ratio, 2),
            "x2": round(x2 * x_ratio, 2), "y2": round(y2 * y_ratio, 2),
            "confidence": round(conf, 3),
            "center_x": round(((x1 + x2) / 2.0) * x_ratio, 2),
            "center_y": round(((y1 + y2) / 2.0) * y_ratio, 2),
        })
    return barrels


# ─── Trajectory Smoothing ─────────────────────────────────────────────────────

def adaptive_max_jump(width, height):
    diagonal = math.hypot(width, height) or 1000.0
    return clamp(diagonal * MAX_JUMP_FRACTION, 80.0, 350.0)

def interpolate_gaps(track_points, max_gap=INTERPOLATION_MAX_GAP):
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
        gap_len = j - i
        if (gap_start >= 0 and j < n
                and result[gap_start] is not None and result[j] is not None
                and gap_len <= max_gap):
            p1, p2 = result[gap_start], result[j]
            for k in range(1, gap_len + 1):
                t = k / float(gap_len + 1)
                result[gap_start + k] = (
                    float(p1[0]) + t * (float(p2[0]) - float(p1[0])),
                    float(p1[1]) + t * (float(p2[1]) - float(p1[1])),
                )
        i = j
    return result

def exponential_smooth(points, alpha=SMOOTHING_ALPHA):
    if not points:
        return []
    smoothed = [points[0]]
    for pt in points[1:]:
        prev = smoothed[-1]
        smoothed.append((
            alpha * float(pt[0]) + (1.0 - alpha) * float(prev[0]),
            alpha * float(pt[1]) + (1.0 - alpha) * float(prev[1]),
        ))
    return smoothed

def dedupe_points(points, min_dist=6.0):
    if not points:
        return []
    deduped = [points[0]]
    for pt in points[1:]:
        if distance(pt, deduped[-1]) > min_dist:
            deduped.append(pt)
    return deduped


# ─── Barrel Identification ────────────────────────────────────────────────────

def identify_barrels(all_barrel_detections, frame_width, frame_height):
    points = []
    for entry in all_barrel_detections:
        for b in entry["barrels"]:
            points.append({"x": float(b["center_x"]), "y": float(b["center_y"]), "confidence": float(b["confidence"])})

    result = {"barrel1": None, "barrel2": None, "barrel3": None}
    geometry = {"top": None, "lower_left": None, "lower_right": None}

    if len(points) < 3:
        result, geometry = _estimate_barrel_positions(frame_width, frame_height)
        return result, geometry, "geometry_estimated"

    sorted_by_y = sorted(points, key=lambda p: p["y"])
    upper_cutoff = sorted_by_y[len(sorted_by_y) // 3]["y"]
    top_candidates = [p for p in points if p["y"] <= upper_cutoff] or sorted_by_y[:3]

    top_x = average_values([p["x"] for p in top_candidates])
    top_y = average_values([p["y"] for p in top_candidates])

    lower_candidates = [p for p in points if p["y"] > (top_y + 10)]
    if len(lower_candidates) < 2:
        lower_candidates = [p for p in points if p not in top_candidates]
    if len(lower_candidates) < 2:
        result, geometry = _estimate_barrel_positions(frame_width, frame_height)
        return result, geometry, "partial_estimated"

    sorted_lower = sorted(lower_candidates, key=lambda p: p["x"])
    split = max(1, len(sorted_lower) // 2)
    left_group = sorted_lower[:split]
    right_group = sorted_lower[split:]

    if not left_group or not right_group:
        median_x = average_values([p["x"] for p in sorted_lower])
        left_group = [p for p in sorted_lower if p["x"] <= median_x]
        right_group = [p for p in sorted_lower if p["x"] > median_x]

    if not left_group or not right_group:
        result, geometry = _estimate_barrel_positions(frame_width, frame_height)
        return result, geometry, "geometry_estimated"

    def cluster(group, role):
        return {
            "center_x": round(average_values([p["x"] for p in group]), 2),
            "center_y": round(average_values([p["y"] for p in group]), 2),
            "detection_count": len(group),
            "average_confidence": round(average_values([p["confidence"] for p in group]), 3),
            "geometry_role": role,
        }

    lower_left = cluster(left_group, "lower_left")
    lower_right = cluster(right_group, "lower_right")
    top_cluster = cluster(top_candidates, "top")
    top_cluster["center_x"] = round(top_x, 2)
    top_cluster["center_y"] = round(top_y, 2)

    geometry["top"] = top_cluster
    geometry["lower_left"] = lower_left
    geometry["lower_right"] = lower_right

    result["barrel1"] = lower_left
    result["barrel2"] = lower_right
    result["barrel3"] = top_cluster

    return result, geometry, "detected"

def _estimate_barrel_positions(frame_width, frame_height):
    w, h = float(frame_width), float(frame_height)
    lower_left = {"center_x": round(w * 0.25, 2), "center_y": round(h * 0.68, 2), "detection_count": 0, "average_confidence": 0.0, "geometry_role": "lower_left"}
    lower_right = {"center_x": round(w * 0.75, 2), "center_y": round(h * 0.68, 2), "detection_count": 0, "average_confidence": 0.0, "geometry_role": "lower_right"}
    top = {"center_x": round(w * 0.50, 2), "center_y": round(h * 0.22, 2), "detection_count": 0, "average_confidence": 0.0, "geometry_role": "top"}
    geometry = {"top": top, "lower_left": lower_left, "lower_right": lower_right}
    result = {"barrel1": lower_left, "barrel2": lower_right, "barrel3": top}
    return result, geometry


# ─── Frame Metrics ────────────────────────────────────────────────────────────

def build_frame_metrics(sampled_frames, identified_barrels):
    metrics = []
    for frame in sampled_frames:
        horse = frame.get("horse_detection")
        horse_center = None
        nearest_barrel = None
        nearest_dist = None
        dist_map = {"barrel1": None, "barrel2": None, "barrel3": None}

        if horse is not None:
            horse_center = (float(horse["tracking_point"][0]), float(horse["tracking_point"][1]))
            for name in ("barrel1", "barrel2", "barrel3"):
                info = identified_barrels.get(name)
                if info:
                    d = distance(horse_center, (float(info["center_x"]), float(info["center_y"])))
                    dist_map[name] = d
            available = [(k, v) for k, v in dist_map.items() if v is not None]
            if available:
                nearest_barrel, nearest_dist = min(available, key=lambda x: x[1])

        metrics.append({
            "frame_index": int(frame["frame_index"]),
            "timestamp_seconds": frame["timestamp_seconds"],
            "horse_detected": horse is not None,
            "horse_center": round_point(horse_center, 2) if horse_center else None,
            "nearest_barrel": nearest_barrel,
            "nearest_barrel_distance_px": round_or_none(nearest_dist, 2),
            "dist_to_barrel1_px": round_or_none(dist_map["barrel1"], 2),
            "dist_to_barrel2_px": round_or_none(dist_map["barrel2"], 2),
            "dist_to_barrel3_px": round_or_none(dist_map["barrel3"], 2),
            "speed_px_per_sec": None,
        })
    return metrics


# ─── Pattern Direction ────────────────────────────────────────────────────────

def detect_pattern_direction(frame_metrics):
    lower_frames = [m for m in frame_metrics if m["horse_detected"] and m["nearest_barrel"] in ("barrel1", "barrel2")]
    if not lower_frames:
        return {"pattern_direction": "left", "actual_to_provisional_map": {"barrel1": "barrel1", "barrel2": "barrel2", "barrel3": "barrel3"}, "reason": "defaulted", "confidence": 0.5, "method": "fallback_left"}
    vote_window = lower_frames[:min(5, len(lower_frames))]
    left_votes = sum(1 for m in vote_window if m["nearest_barrel"] == "barrel1")
    right_votes = sum(1 for m in vote_window if m["nearest_barrel"] == "barrel2")
    if right_votes > left_votes:
        return {"pattern_direction": "right", "actual_to_provisional_map": {"barrel1": "barrel2", "barrel2": "barrel1", "barrel3": "barrel3"}, "reason": "early approach favored right side", "confidence": 0.80, "method": "early_lower_votes"}
    return {"pattern_direction": "left", "actual_to_provisional_map": {"barrel1": "barrel1", "barrel2": "barrel2", "barrel3": "barrel3"}, "reason": "early approach favored left side", "confidence": 0.80, "method": "early_lower_votes"}

def remap_barrel_keyed_dict(provisional_dict, actual_to_provisional_map):
    return {actual: provisional_dict.get(actual_to_provisional_map.get(actual)) for actual in ("barrel1", "barrel2", "barrel3")}

def remap_frame_metric_labels(metrics, actual_to_provisional_map):
    provisional_to_actual = {v: k for k, v in actual_to_provisional_map.items()}
    remapped = []
    for m in metrics:
        nm = dict(m)
        nm["dist_to_barrel1_px"] = m.get(f"dist_to_{actual_to_provisional_map['barrel1']}_px")
        nm["dist_to_barrel2_px"] = m.get(f"dist_to_{actual_to_provisional_map['barrel2']}_px")
        nm["dist_to_barrel3_px"] = m.get(f"dist_to_{actual_to_provisional_map['barrel3']}_px")
        prov = m.get("nearest_barrel")
        nm["nearest_barrel"] = provisional_to_actual.get(prov) if prov else None
        remapped.append(nm)
    return remapped


# ─── Apex / Turn Detection ────────────────────────────────────────────────────

def find_barrel_apex(barrel_name, frame_metrics, min_approach_frames=2):
    dist_key = f"dist_to_{barrel_name}_px"
    valid = [m for m in frame_metrics if m["horse_detected"] and m.get(dist_key) is not None]
    if len(valid) < min_approach_frames:
        return None
    distances = [m[dist_key] for m in valid]
    smoothed = list(distances)
    for i in range(1, len(smoothed) - 1):
        smoothed[i] = (distances[i - 1] + distances[i] + distances[i + 1]) / 3.0
    min_idx = smoothed.index(min(smoothed))
    if min_idx == 0 and len(valid) > 1:
        min_idx = smoothed[1:].index(min(smoothed[1:])) + 1
    apex = valid[min_idx]
    return {
        "barrel_name": barrel_name,
        "apex_frame": int(apex["frame_index"]),
        "apex_timestamp_seconds": apex["timestamp_seconds"],
        "min_distance_px": round_or_none(apex[dist_key], 2),
        "smoothed_min_distance_px": round_or_none(smoothed[min_idx], 2),
    }

def build_turns(frame_metrics):
    return {
        "barrel1": find_barrel_apex("barrel1", frame_metrics),
        "barrel2": find_barrel_apex("barrel2", frame_metrics),
        "barrel3": find_barrel_apex("barrel3", frame_metrics),
    }

def enforce_turn_order(turns):
    b1 = turns.get("barrel1")
    b2 = turns.get("barrel2")
    b3 = turns.get("barrel3")
    fixed = dict(turns)
    t1 = b1["apex_timestamp_seconds"] if b1 else None
    t2 = b2["apex_timestamp_seconds"] if b2 else None
    t3 = b3["apex_timestamp_seconds"] if b3 else None
    if t1 and t2 and t2 <= t1:
        fixed["barrel2"] = None
        t2 = None
    if t2 and t3 and t3 <= t2:
        fixed["barrel3"] = None
    elif t1 and t3 and not t2 and t3 <= t1:
        fixed["barrel3"] = None
    return fixed


# ─── Splits ───────────────────────────────────────────────────────────────────

def build_splits(turns, frame_metrics, total_run_time_seconds=None):
    valid = [m for m in frame_metrics if m["horse_detected"] and m["timestamp_seconds"] is not None]
    empty = {"start_to_barrel1_seconds": None, "barrel1_to_barrel2_seconds": None, "barrel2_to_barrel3_seconds": None, "barrel3_to_home_seconds": None, "splits_method": "no_data"}
    if not valid:
        return empty

    b1 = turns.get("barrel1")
    b2 = turns.get("barrel2")
    b3 = turns.get("barrel3")
    video_start = valid[0]["timestamp_seconds"]
    t1 = (b1["apex_timestamp_seconds"] - video_start) if b1 else None
    t2 = (b2["apex_timestamp_seconds"] - video_start) if b2 else None
    t3 = (b3["apex_timestamp_seconds"] - video_start) if b3 else None

    def safe(v):
        if v is None:
            return None
        if v < MIN_SPLIT_SECONDS or v > MAX_SPLIT_SECONDS:
            return None
        return round_or_none(v, 3)

    if t3 is None:
        return {"start_to_barrel1_seconds": safe(t1), "barrel1_to_barrel2_seconds": safe((t2 - t1) if t1 and t2 else None), "barrel2_to_barrel3_seconds": None, "barrel3_to_home_seconds": None, "splits_method": "partial_no_barrel3"}

    has_valid_run_time = (total_run_time_seconds is not None and MIN_RUN_TIME <= total_run_time_seconds <= MAX_RUN_TIME)

    if has_valid_run_time:
        total = total_run_time_seconds
        t_home = total - t3
        if 0.5 <= t_home <= 8.0:
            s1 = safe(t1)
            s2 = safe((t2 - t1) if t1 and t2 else None)
            s3 = safe((t3 - t2) if t2 else None)
            s4 = safe(t_home)
            known = [x for x in [s1, s2, s3, s4] if x is not None]
            if known and abs(sum(known) - total) < total * 0.25:
                return {"start_to_barrel1_seconds": s1, "barrel1_to_barrel2_seconds": s2, "barrel2_to_barrel3_seconds": s3, "barrel3_to_home_seconds": s4, "splits_method": "anchored_to_run_time"}
        if t3 > 0:
            estimated_fraction_at_b3 = 0.78
            scale = clamp((total * estimated_fraction_at_b3) / t3, 0.5, 2.5)
            s1 = safe(t1 * scale if t1 else None)
            s2 = safe((t2 - t1) * scale if t1 and t2 else None)
            s3 = safe((t3 - t2) * scale if t2 else None)
            accounted = sum(x for x in [s1, s2, s3] if x is not None)
            s4 = safe(total - accounted)
            if s4 is not None and s4 < 0.5:
                s4 = 0.5
            return {"start_to_barrel1_seconds": s1, "barrel1_to_barrel2_seconds": s2, "barrel2_to_barrel3_seconds": s3, "barrel3_to_home_seconds": s4, "splits_method": "proportional_scaled"}

    return {"start_to_barrel1_seconds": safe(t1), "barrel1_to_barrel2_seconds": safe((t2 - t1) if t1 and t2 else None), "barrel2_to_barrel3_seconds": safe((t3 - t2) if t2 else None), "barrel3_to_home_seconds": None, "splits_method": "raw_video_timestamps"}


# ─── Highlights & Insights ────────────────────────────────────────────────────

def build_highlights(frame_metrics):
    barrel_distances = {"barrel1": [], "barrel2": [], "barrel3": []}
    for m in frame_metrics:
        for name in ("barrel1", "barrel2", "barrel3"):
            d = m.get(f"dist_to_{name}_px")
            if d is not None:
                barrel_distances[name].append(float(d))

    avg_distances = {k: (sum(v) / len(v)) if v else None for k, v in barrel_distances.items()}
    available = [(k, v) for k, v in avg_distances.items() if v is not None]
    best_barrel = min(available, key=lambda x: x[1])[0] if available else None
    weakest_barrel = max(available, key=lambda x: x[1])[0] if available else None

    label_map = {"barrel1": "1st", "barrel2": "2nd", "barrel3": "3rd"}
    focus_map = {
        "barrel1": "Work your approach angle and rate point to the first barrel",
        "barrel2": "Stay tighter through the second barrel — focus on your pocket",
        "barrel3": "Carry a cleaner line into and out of the third barrel",
    }

    return {
        "best_barrel": label_map.get(best_barrel),
        "best_turn": label_map.get(best_barrel),
        "focus_next": focus_map.get(weakest_barrel, "Work on consistency across all three barrels"),
        "weakest_barrel": label_map.get(weakest_barrel),
    }


def build_insights(tracking_quality, barrel_detection_summary, pattern_direction_info, splits, highlights, speed_summary=None):
    insights = []
    method = splits.get("splits_method", "")
    detection_rate = tracking_quality.get("horse_detection_rate", 1.0)

    if detection_rate < 0.5:
        insights.append(f"Horse detected in only {round(detection_rate * 100)}% of frames — a wider camera angle or better lighting will improve accuracy.")

    if barrel_detection_summary.get("detected_frame_count", 0) < 3:
        insights.append("Barrel positions were hard to confirm. A wider shot keeping all three barrels visible will improve accuracy.")

    if speed_summary:
        slowest = speed_summary.get("slowest_section_label")
        if slowest:
            insights.append(f"Slowest section: {slowest} — this is where the most time is being lost.")

    split_messages = {
        "raw_video_timestamps": "No run time entered — splits are raw video estimates. Enter your run time for better accuracy.",
        "partial_no_barrel3": "Third barrel apex not detected — barrel3-to-home unavailable.",
    }
    if method in split_messages:
        insights.append(split_messages[method])

    return insights[:6]


def summarize_barrel_detections(all_barrel_detections):
    centers = []
    for entry in all_barrel_detections:
        for b in entry["barrels"]:
            centers.append({"center_x": b["center_x"], "center_y": b["center_y"], "confidence": b["confidence"], "frame_index": entry["frame_index"]})
    if not centers:
        return {"detected_frame_count": 0, "total_barrel_boxes": 0, "average_barrels_per_detected_frame": 0.0, "top_barrel_centers": []}
    detected_frame_count = sum(1 for e in all_barrel_detections if e["barrels"])
    total = len(centers)
    top = sorted(centers, key=lambda c: c["confidence"], reverse=True)[:8]
    return {"detected_frame_count": detected_frame_count, "total_barrel_boxes": total, "average_barrels_per_detected_frame": round(total / max(detected_frame_count, 1), 3), "top_barrel_centers": top}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_json_str = sys.argv[2] if len(sys.argv) > 2 else "{}"

    if not video_path:
        fail("No video path provided.")
        return
    if not os.path.exists(video_path):
        fail("Video file does not exist.", {"video_path": video_path})
        return
    if not os.path.exists(BARREL_MODEL_PATH):
        fail("Barrel model not found.", {"barrel_model_path": BARREL_MODEL_PATH})
        return
    if not os.path.exists(HORSE_MODEL_PATH):
        fail("Horse model not found.", {"horse_model_path": HORSE_MODEL_PATH})
        return

    try:
        run_data = json.loads(run_json_str) if run_json_str else {}
    except Exception:
        run_data = {}

    total_run_time_seconds = None
    try:
        t = float(run_data.get("time", 0) or 0)
        if MIN_RUN_TIME <= t <= MAX_RUN_TIME:
            total_run_time_seconds = t
    except Exception:
        pass

    try:
        log_err("Loading models...")
        barrel_model = load_model(BARREL_MODEL_PATH)
        horse_model = load_model(HORSE_MODEL_PATH)
    except Exception as e:
        fail("Failed to load YOLO model.", {"details": str(e)})
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        fail("Could not open video.", {"video_path": video_path})
        return

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0

        if frame_count <= 0 or original_width <= 0 or original_height <= 0:
            fail("Video metadata unreadable.")
            return

        log_err(f"Video: frames={frame_count}, fps={round(fps,2)}, {original_width}x{original_height}, duration={round(duration,2)}s")

        max_jump = adaptive_max_jump(original_width, original_height)
        base_indices = build_base_sample_indices(frame_count, fps)
        log_err(f"Base samples: {len(base_indices)}")

        sampled_frames = []
        all_barrel_detections = []
        horse_detected_count = 0
        read_success_count = 0
        rejected_jump_count = 0
        raw_trajectory_points = []
        accepted_points = []
        previous_accepted_point = None

        # ── Pass 1: Base sampling ─────────────────────────────────────────────
        for idx, target_frame in enumerate(base_indices):
            frame = read_frame_at(cap, target_frame)
            horse_detection = None
            barrel_detections = []
            rejection_reason = None
            percent = (target_frame / max(frame_count - 1, 1)) if frame_count > 1 else 0.0
            timestamp_seconds = round(target_frame / fps, 3) if fps > 0 else None

            if frame is not None:
                read_success_count += 1
                inf_frame, xr, yr = resize_for_inference(frame)

                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    horse_results = horse_model.predict(source=inf_frame, conf=HORSE_CONFIDENCE_THRESHOLD, classes=[HORSE_CLASS_ID], verbose=False)

                candidates = detect_horse_candidates(horse_results[0], xr, yr) if horse_results else []
                best = choose_best_horse(candidates, previous_accepted_point, original_width, original_height)

                barrel_detections = detect_barrels(inf_frame, barrel_model, xr, yr)
                all_barrel_detections.append({"frame_index": int(target_frame), "timestamp_seconds": timestamp_seconds, "barrels": barrel_detections})

                if best is not None:
                    horse_detected_count += 1
                    horse_detection = {"confidence": best["confidence"], "bbox": best["bbox"], "center": best["center"], "tracking_point": round_point(best["tracking_point"], 2)}
                    current_point = best["tracking_point"]
                    raw_trajectory_points.append(current_point)
                    jump = distance(previous_accepted_point, current_point) if previous_accepted_point else 0
                    if previous_accepted_point is None or jump <= max_jump * 1.12:
                        accepted_points.append(current_point)
                        previous_accepted_point = current_point
                    else:
                        rejected_jump_count += 1
                        rejection_reason = f"jump_rejected_{round(jump,1)}px"

            sampled_frames.append({
                "percent": round(percent, 4), "frame_index": int(target_frame),
                "timestamp_seconds": timestamp_seconds, "read_success": frame is not None,
                "horse_detection": horse_detection, "barrel_detections": barrel_detections,
                "barrel_detection_count": len(barrel_detections), "rejection_reason": rejection_reason,
            })

        # ── Preliminary identification for dense pass ─────────────────────────
        prov_barrels, prov_geometry, _ = identify_barrels(all_barrel_detections, original_width, original_height)
        prov_metrics = build_frame_metrics(sampled_frames, prov_barrels)
        direction_info = detect_pattern_direction(prov_metrics)
        a2p = direction_info["actual_to_provisional_map"]
        prov_barrels_remapped = remap_barrel_keyed_dict(prov_barrels, a2p)
        prov_metrics_remapped = remap_frame_metric_labels(prov_metrics, a2p)
        prov_turns = enforce_turn_order(build_turns(prov_metrics_remapped))
        apex_timestamps = [t["apex_timestamp_seconds"] for t in prov_turns.values() if t and t.get("apex_timestamp_seconds")]

        # ── Pass 2: Dense sampling around apexes ──────────────────────────────
        if apex_timestamps:
            dense_indices = build_dense_indices_around_apexes(apex_timestamps, fps, frame_count, base_indices)
            new_indices = sorted(set(dense_indices) - set(base_indices))
            log_err(f"Dense pass: +{len(new_indices)} frames around apex zones")

            for target_frame in new_indices:
                frame = read_frame_at(cap, target_frame)
                timestamp_seconds = round(target_frame / fps, 3) if fps > 0 else None
                percent = (target_frame / max(frame_count - 1, 1)) if frame_count > 1 else 0.0
                horse_detection = None
                barrel_detections = []

                if frame is not None:
                    read_success_count += 1
                    inf_frame, xr, yr = resize_for_inference(frame)

                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        horse_results = horse_model.predict(source=inf_frame, conf=HORSE_CONFIDENCE_THRESHOLD, classes=[HORSE_CLASS_ID], verbose=False)

                    candidates = detect_horse_candidates(horse_results[0], xr, yr) if horse_results else []
                    best = choose_best_horse(candidates, previous_accepted_point, original_width, original_height)
                    barrel_detections = detect_barrels(inf_frame, barrel_model, xr, yr)
                    all_barrel_detections.append({"frame_index": int(target_frame), "timestamp_seconds": timestamp_seconds, "barrels": barrel_detections})

                    if best is not None:
                        horse_detected_count += 1
                        horse_detection = {"confidence": best["confidence"], "bbox": best["bbox"], "center": best["center"], "tracking_point": round_point(best["tracking_point"], 2)}
                        current_point = best["tracking_point"]
                        jump = distance(previous_accepted_point, current_point) if previous_accepted_point else 0
                        if previous_accepted_point is None or jump <= max_jump * 1.12:
                            accepted_points.append(current_point)
                            previous_accepted_point = current_point

                sampled_frames.append({
                    "percent": round(percent, 4), "frame_index": int(target_frame),
                    "timestamp_seconds": timestamp_seconds, "read_success": frame is not None,
                    "horse_detection": horse_detection, "barrel_detections": barrel_detections,
                    "barrel_detection_count": len(barrel_detections), "rejection_reason": None, "dense_pass": True,
                })

            sampled_frames.sort(key=lambda f: f["frame_index"])
            all_barrel_detections.sort(key=lambda e: e["frame_index"])

        # ── Final pass with all data ───────────────────────────────────────────
        final_barrels, final_geometry, barrel_id_method = identify_barrels(all_barrel_detections, original_width, original_height)
        final_barrels = remap_barrel_keyed_dict(final_barrels, a2p)
        final_metrics_raw = remap_frame_metric_labels(build_frame_metrics(sampled_frames, final_barrels), a2p)
        final_metrics = calculate_speed_profile(final_metrics_raw, fps)
        final_turns = enforce_turn_order(build_turns(final_metrics))
        splits = build_splits(final_turns, final_metrics, total_run_time_seconds)
        speed_summary = build_speed_summary(final_metrics, final_turns, fps)
        highlights = build_highlights(final_metrics)

        # ── Trajectory ────────────────────────────────────────────────────────
        all_track_points = [tuple(f["horse_detection"]["tracking_point"]) if f.get("horse_detection") else None for f in sampled_frames]
        interpolated = interpolate_gaps(all_track_points, INTERPOLATION_MAX_GAP)
        smoothed_points = exponential_smooth([p for p in interpolated if p is not None], SMOOTHING_ALPHA)
        smoothed_points = dedupe_points(smoothed_points, min_dist=6.0)

        # ── Quality summary ───────────────────────────────────────────────────
        total_sampled = len(sampled_frames)
        tracking_quality = {
            "sampled_frame_count": total_sampled,
            "read_success_count": read_success_count,
            "horse_detected_count": horse_detected_count,
            "rejected_jump_count": rejected_jump_count,
            "max_jump_threshold_px": round(float(max_jump), 2),
            "read_success_rate": round(read_success_count / max(total_sampled, 1), 4),
            "horse_detection_rate": round(horse_detected_count / max(read_success_count, 1), 4),
            "barrel_id_method": barrel_id_method,
            "dense_pass_used": bool(apex_timestamps),
        }

        barrel_detection_summary = summarize_barrel_detections(all_barrel_detections)
        direction = direction_info.get("pattern_direction", "left")
        insights = build_insights(tracking_quality, barrel_detection_summary, direction_info, splits, highlights, speed_summary)

        # ── Output ────────────────────────────────────────────────────────────
        emit_json({
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
            "smoothed_path_points": [round_point(p, 2) for p in smoothed_points],
            "tracking_quality": tracking_quality,
            "barrel_detection_summary": barrel_detection_summary,
            "barrel_geometry": final_geometry,
            "pattern_direction": direction,
            "pattern_direction_info": direction_info,
            "identified_barrels": final_barrels,
            "turns": final_turns,
            "splits": splits,
            "speed_summary": speed_summary,
            "highlights": highlights,
            "frame_metrics": final_metrics,
            "insights": insights,
        })

    except Exception as e:
        log_err("Python analysis crashed:", str(e))
        log_err(traceback.format_exc())
        emit_json({"ok": False, "error": "Python analysis crashed", "details": str(e), "traceback": traceback.format_exc(), "video_path": video_path})
    finally:
        cap.release()


if __name__ == "__main__":
    main()