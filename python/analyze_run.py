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

# Sampling — base pass
TARGET_SAMPLE_FPS = 4.0        # IMPROVED: doubled from 2.0 for more data
MIN_SAMPLED_FRAMES = 16
MAX_SAMPLED_FRAMES = 48        # IMPROVED: doubled from 24

# Dense sampling around barrel apex zones
DENSE_SAMPLE_FPS = 12.0        # IMPROVED: increased from 8.0
DENSE_WINDOW_SECONDS = 1.5     # IMPROVED: wider window from 1.2

# Inference size — IMPROVED: larger for better detection
MAX_INFERENCE_WIDTH = 768
MAX_INFERENCE_HEIGHT = 432

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

# Turn quality grading thresholds (pixels)
TURN_GRADE_A_PX = 25.0   # Excellent — very tight to barrel
TURN_GRADE_B_PX = 50.0   # Good
TURN_GRADE_C_PX = 80.0   # Acceptable
TURN_GRADE_D_PX = 120.0  # Wide — losing time here

# Approach angle ideal range (degrees)
IDEAL_APPROACH_ANGLE_MIN = 20.0
IDEAL_APPROACH_ANGLE_MAX = 40.0

# Barrel knockdown detection
BARREL_MOVEMENT_THRESHOLD_PX = 30.0  # pixels of movement = potential knockdown


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


# ─── NEW: Speed Calculation ───────────────────────────────────────────────────

def calculate_speed_profile(frame_metrics, fps):
    """
    Calculate horse speed (pixels/second) at each frame.
    Returns frame_metrics enriched with speed data.
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


# ─── NEW: Approach Angle Calculation ─────────────────────────────────────────

def calculate_approach_angle(frame_metrics, barrel_name, barrel_info, fps):
    """
    Calculate the angle (degrees) at which the horse approaches each barrel.
    Uses frames BEFORE the apex to determine direction of travel.
    
    Ideal approach: 20-40 degrees (slight curve into barrel)
    Straight-on (< 15 degrees): horse will blow past the pocket
    Too wide (> 50 degrees): horse losing momentum on sharp turn
    """
    if not barrel_info:
        return None
    
    dist_key = f"dist_to_{barrel_name}_px"
    barrel_x = float(barrel_info.get("center_x", 0))
    barrel_y = float(barrel_info.get("center_y", 0))
    
    # Find frames approaching the barrel (distance decreasing)
    valid = [m for m in frame_metrics if m.get("horse_center") and m.get(dist_key) is not None]
    if len(valid) < 4:
        return None
    
    # Find the apex frame index
    distances = [m[dist_key] for m in valid]
    min_idx = distances.index(min(distances))
    
    # Take 3-6 frames before apex for approach direction
    approach_start = max(0, min_idx - 6)
    approach_end = max(0, min_idx - 1)
    
    if approach_end <= approach_start:
        return None
    
    approach_frames = valid[approach_start:approach_end + 1]
    if len(approach_frames) < 2:
        return None
    
    # Calculate direction vector of approach
    first = approach_frames[0]["horse_center"]
    last = approach_frames[-1]["horse_center"]
    
    dx = float(last[0]) - float(first[0])
    dy = float(last[1]) - float(first[1])
    
    if abs(dx) < 1.0 and abs(dy) < 1.0:
        return None
    
    # Vector from last approach point to barrel
    to_barrel_x = barrel_x - float(last[0])
    to_barrel_y = barrel_y - float(last[1])
    
    # Angle between movement direction and barrel direction
    dot = dx * to_barrel_x + dy * to_barrel_y
    mag1 = math.hypot(dx, dy)
    mag2 = math.hypot(to_barrel_x, to_barrel_y)
    
    if mag1 < 1.0 or mag2 < 1.0:
        return None
    
    cos_angle = clamp(dot / (mag1 * mag2), -1.0, 1.0)
    angle_deg = math.degrees(math.acos(cos_angle))
    
    return round(angle_deg, 1)


def grade_approach_angle(angle_deg):
    """Grade the approach angle."""
    if angle_deg is None:
        return "unknown"
    if IDEAL_APPROACH_ANGLE_MIN <= angle_deg <= IDEAL_APPROACH_ANGLE_MAX:
        return "ideal"
    elif angle_deg < IDEAL_APPROACH_ANGLE_MIN:
        return "too_straight"  # running past the pocket
    else:
        return "too_wide"  # losing momentum on sharp entry


# ─── NEW: Turn Tightness Grading ──────────────────────────────────────────────

def grade_turn_tightness(min_distance_px):
    """
    Convert minimum approach distance to a letter grade.
    Smaller distance = tighter turn = better.
    """
    if min_distance_px is None:
        return {"grade": "unknown", "label": "No data", "coaching_note": "Barrel not detected clearly"}
    
    d = float(min_distance_px)
    if d <= TURN_GRADE_A_PX:
        return {"grade": "A", "label": "Excellent", "coaching_note": "Very tight to the barrel — good pocket"}
    elif d <= TURN_GRADE_B_PX:
        return {"grade": "B", "label": "Good", "coaching_note": "Solid turn, slight room to tighten"}
    elif d <= TURN_GRADE_C_PX:
        return {"grade": "C", "label": "Acceptable", "coaching_note": "Running wide — tightening this turn will save time"}
    elif d <= TURN_GRADE_D_PX:
        return {"grade": "D", "label": "Wide", "coaching_note": "Significantly wide — major time being lost here"}
    else:
        return {"grade": "F", "label": "Very Wide", "coaching_note": "Extremely wide turn — this barrel is costing serious time"}


# ─── NEW: Exit Acceleration Calculation ───────────────────────────────────────

def calculate_exit_acceleration(frame_metrics, barrel_name, apex_timestamp):
    """
    Compare horse speed AT apex vs speed 10+ frames after apex.
    Positive = horse accelerated out (good)
    Negative = horse decelerated/drifted out (bad — losing time)
    """
    if apex_timestamp is None:
        return None
    
    dist_key = f"dist_to_{barrel_name}_px"
    valid = [m for m in frame_metrics 
             if m.get("horse_center") and m.get("speed_px_per_sec") is not None
             and m.get("timestamp_seconds") is not None]
    
    if not valid:
        return None
    
    # Speed at apex (within 0.2s)
    at_apex = [m for m in valid if abs(m["timestamp_seconds"] - apex_timestamp) <= 0.2]
    # Speed after apex (0.3s to 1.0s after)
    after_apex = [m for m in valid 
                  if 0.3 <= (m["timestamp_seconds"] - apex_timestamp) <= 1.0]
    
    if not at_apex or not after_apex:
        return None
    
    apex_speed = average_values([m["speed_px_per_sec"] for m in at_apex])
    exit_speed = average_values([m["speed_px_per_sec"] for m in after_apex])
    
    if apex_speed is None or exit_speed is None or apex_speed < 1.0:
        return None
    
    acceleration_ratio = exit_speed / apex_speed
    
    return {
        "apex_speed_px_per_sec": round(apex_speed, 1),
        "exit_speed_px_per_sec": round(exit_speed, 1),
        "acceleration_ratio": round(acceleration_ratio, 3),
        "drove_out": acceleration_ratio >= 1.05,
        "drifted": acceleration_ratio < 0.90,
        "coaching_note": (
            "Good drive out of the barrel" if acceleration_ratio >= 1.05
            else "Horse drifted — not driving forward out of the turn" if acceleration_ratio < 0.90
            else "Neutral exit — consistent speed through turn"
        )
    }


# ─── NEW: Barrel Knockdown Detection ─────────────────────────────────────────

def detect_barrel_knockdowns(all_barrel_detections, identified_barrels):
    """
    Detect potential barrel knockdowns by tracking barrel position stability.
    If a barrel moves significantly between frames, flag it as potentially knocked.
    
    Returns dict of barrel_name -> knockdown info
    """
    results = {
        "barrel1": {"knocked": False, "confidence": 0.0, "evidence": []},
        "barrel2": {"knocked": False, "confidence": 0.0, "evidence": []},
        "barrel3": {"knocked": False, "confidence": 0.0, "evidence": []},
    }
    
    if not identified_barrels:
        return results
    
    # Build per-barrel position history from detections
    barrel_positions = {"barrel1": [], "barrel2": [], "barrel3": []}
    
    for entry in all_barrel_detections:
        frame_idx = entry["frame_index"]
        ts = entry.get("timestamp_seconds")
        
        for detected_barrel in entry.get("barrels", []):
            cx = detected_barrel["center_x"]
            cy = detected_barrel["center_y"]
            
            # Match detected barrel to identified barrel (closest center)
            best_match = None
            best_dist = float("inf")
            
            for barrel_name, barrel_info in identified_barrels.items():
                if barrel_info is None:
                    continue
                d = distance((cx, cy), (barrel_info["center_x"], barrel_info["center_y"]))
                if d < best_dist and d < 100:  # max 100px from known barrel position
                    best_dist = d
                    best_match = barrel_name
            
            if best_match:
                barrel_positions[best_match].append({
                    "frame_index": frame_idx,
                    "timestamp_seconds": ts,
                    "cx": cx,
                    "cy": cy,
                    "dist_from_baseline": best_dist,
                })
    
    # Check for significant movement in each barrel's position
    for barrel_name, positions in barrel_positions.items():
        if len(positions) < 3:
            continue
        
        # Sort by frame
        positions.sort(key=lambda p: p["frame_index"])
        
        # Calculate median position as baseline
        median_cx = sorted([p["cx"] for p in positions])[len(positions) // 2]
        median_cy = sorted([p["cy"] for p in positions])[len(positions) // 2]
        
        # Look for frames where barrel moved significantly from median
        outliers = []
        for p in positions:
            d = distance((p["cx"], p["cy"]), (median_cx, median_cy))
            if d > BARREL_MOVEMENT_THRESHOLD_PX:
                outliers.append({
                    "frame_index": p["frame_index"],
                    "timestamp_seconds": p["timestamp_seconds"],
                    "movement_px": round(d, 1),
                })
        
        if outliers:
            # Check if outliers happen AFTER horse was near the barrel
            # (not just detection noise early in the run)
            confidence = min(0.9, len(outliers) * 0.3)
            results[barrel_name] = {
                "knocked": confidence >= 0.3,
                "confidence": round(confidence, 2),
                "evidence": outliers[:3],  # top 3 evidence frames
                "coaching_note": f"Barrel may have been disturbed — {len(outliers)} detection(s) showed movement",
            }
    
    return results


# ─── NEW: Speed Summary ───────────────────────────────────────────────────────

def build_speed_summary(frame_metrics, turns, fps):
    """
    Build a comprehensive speed summary for the entire run.
    Identifies fastest and slowest sections.
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
    
    # Find which section (between barrels) was slowest
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


# ─── NEW: Comprehensive Barrel Metrics ───────────────────────────────────────

def build_comprehensive_barrel_metrics(frame_metrics, identified_barrels, turns, all_barrel_detections, fps):
    """
    Build rich per-barrel coaching metrics combining:
    - Turn tightness grade
    - Approach angle
    - Exit acceleration
    - Speed at apex
    - Barrel knockdown
    """
    knockdowns = detect_barrel_knockdowns(all_barrel_detections, identified_barrels)
    metrics = {}
    
    for barrel_name in ("barrel1", "barrel2", "barrel3"):
        barrel_info = identified_barrels.get(barrel_name)
        turn = turns.get(barrel_name)
        
        # Tightness
        min_dist = turn.get("min_distance_px") if turn else None
        tightness = grade_turn_tightness(min_dist)
        
        # Approach angle
        angle = calculate_approach_angle(frame_metrics, barrel_name, barrel_info, fps)
        angle_grade = grade_approach_angle(angle)
        
        # Exit acceleration
        apex_ts = turn.get("apex_timestamp_seconds") if turn else None
        exit_accel = calculate_exit_acceleration(frame_metrics, barrel_name, apex_ts)
        
        # Speed at apex
        apex_speed = None
        if apex_ts is not None:
            near_apex = [m for m in frame_metrics 
                        if m.get("speed_px_per_sec") is not None
                        and m.get("timestamp_seconds") is not None
                        and abs(m["timestamp_seconds"] - apex_ts) <= 0.25]
            if near_apex:
                apex_speed = average_values([m["speed_px_per_sec"] for m in near_apex])
        
        # Knockdown
        knockdown = knockdowns.get(barrel_name, {})
        
        # Overall barrel score (for AI to reference)
        score_components = []
        if tightness["grade"] in ("A", "B"):
            score_components.append("tight_turn")
        elif tightness["grade"] in ("D", "F"):
            score_components.append("wide_turn")
        
        if angle_grade == "ideal":
            score_components.append("good_approach")
        elif angle_grade == "too_straight":
            score_components.append("straight_approach")
        elif angle_grade == "too_wide":
            score_components.append("wide_approach")
        
        if exit_accel and exit_accel.get("drove_out"):
            score_components.append("good_exit_drive")
        elif exit_accel and exit_accel.get("drifted"):
            score_components.append("drifted_exit")
        
        label_map = {"barrel1": "First", "barrel2": "Second", "barrel3": "Third"}
        
        metrics[barrel_name] = {
            "barrel_label": label_map.get(barrel_name, barrel_name),
            "detected": barrel_info is not None,
            "turn_tightness": {
                "min_distance_px": round_or_none(min_dist, 1),
                "grade": tightness["grade"],
                "label": tightness["label"],
                "coaching_note": tightness["coaching_note"],
            },
            "approach": {
                "angle_degrees": angle,
                "grade": angle_grade,
                "ideal_range_degrees": f"{IDEAL_APPROACH_ANGLE_MIN}-{IDEAL_APPROACH_ANGLE_MAX}",
                "coaching_note": (
                    "Good approach angle" if angle_grade == "ideal"
                    else "Coming in too straight — will run past the pocket" if angle_grade == "too_straight"
                    else "Coming in too wide — losing momentum" if angle_grade == "too_wide"
                    else "Could not calculate"
                ),
            },
            "exit_drive": exit_accel,
            "speed_at_apex_px_per_sec": round(apex_speed, 1) if apex_speed else None,
            "potential_knockdown": knockdown.get("knocked", False),
            "knockdown_confidence": knockdown.get("confidence", 0.0),
            "knockdown_note": knockdown.get("coaching_note") if knockdown.get("knocked") else None,
            "summary_tags": score_components,
        }
    
    return metrics


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
            "speed_px_per_sec": None,  # filled in later
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

def build_highlights(frame_metrics, barrel_metrics=None):
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

    # Use barrel metrics grades if available
    if barrel_metrics:
        grades = {name: barrel_metrics[name]["turn_tightness"]["grade"] for name in ("barrel1", "barrel2", "barrel3") if barrel_metrics.get(name)}
        worst_grade_barrel = None
        grade_order = ["F", "D", "C", "B", "A", "unknown"]
        for grade in grade_order:
            for name, g in grades.items():
                if g == grade:
                    worst_grade_barrel = name
                    break
            if worst_grade_barrel:
                break
        if worst_grade_barrel:
            weakest_barrel = worst_grade_barrel

    return {
        "best_barrel": label_map.get(best_barrel),
        "best_turn": label_map.get(best_barrel),
        "focus_next": focus_map.get(weakest_barrel, "Work on consistency across all three barrels"),
        "weakest_barrel": label_map.get(weakest_barrel),
    }


def build_insights(tracking_quality, barrel_detection_summary, pattern_direction_info, splits, highlights, barrel_metrics=None, speed_summary=None):
    insights = []
    method = splits.get("splits_method", "")
    detection_rate = tracking_quality.get("horse_detection_rate", 1.0)

    if detection_rate < 0.5:
        insights.append(f"Horse detected in only {round(detection_rate * 100)}% of frames — a wider camera angle or better lighting will improve accuracy.")
    
    if barrel_detection_summary.get("detected_frame_count", 0) < 3:
        insights.append("Barrel positions were hard to confirm. A wider shot keeping all three barrels visible will improve accuracy.")

    # Speed insights
    if speed_summary:
        slowest = speed_summary.get("slowest_section_label")
        if slowest:
            insights.append(f"Slowest section: {slowest} — this is where the most time is being lost.")

    # Barrel-specific insights from grades
    if barrel_metrics:
        for barrel_name in ("barrel1", "barrel2", "barrel3"):
            bm = barrel_metrics.get(barrel_name, {})
            label = bm.get("barrel_label", barrel_name)
            grade = bm.get("turn_tightness", {}).get("grade")
            approach = bm.get("approach", {}).get("grade")
            knocked = bm.get("potential_knockdown", False)
            
            if knocked:
                insights.append(f"{label} barrel may have been disturbed — possible knockdown detected.")
            elif grade in ("D", "F"):
                insights.append(f"{label} barrel turn is wide (grade {grade}) — tightening this turn will save time.")
            
            if approach == "too_straight":
                insights.append(f"Approaching the {label.lower()} barrel too straight — will run past the pocket.")

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


# ─── Overlay Drawing ──────────────────────────────────────────────────────────

def draw_overlay(frame, horse_detection, barrel_detections, frame_index, timestamp_seconds):
    overlay = frame.copy()
    for b in (barrel_detections or []):
        cv2.rectangle(overlay, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), (0, 255, 255), 2)
        cv2.putText(overlay, f"barrel {b['confidence']:.2f}", (int(b["x1"]), max(20, int(b["y1"]) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2, cv2.LINE_AA)
    if horse_detection:
        x1, y1, x2, y2 = [int(v) for v in horse_detection["bbox"]]
        cx, cy = [int(v) for v in horse_detection["tracking_point"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(overlay, f"horse {horse_detection['confidence']:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"frame {frame_index}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if timestamp_seconds is not None:
        cv2.putText(overlay, f"{timestamp_seconds:.2f}s", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


# ─── Ideal Path ───────────────────────────────────────────────────────────────

def build_left_first_ideal_waypoints():
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

def resample_polyline(points, num_samples=100):
    if not points or len(points) == 1:
        return [points[0]] * num_samples if points else []
    deduped = [points[0]]
    for pt in points[1:]:
        if distance(pt, deduped[-1]) > 0.001:
            deduped.append(pt)
    if len(deduped) == 1:
        return [deduped[0]] * num_samples
    cumulative = [0.0]
    for i in range(1, len(deduped)):
        cumulative.append(cumulative[-1] + distance(deduped[i - 1], deduped[i]))
    total = cumulative[-1]
    if total <= 1e-9:
        return [deduped[0]] * num_samples
    samples = []
    for i in range(num_samples):
        target = (total * i) / max(num_samples - 1, 1)
        seg = 0
        while seg < len(cumulative) - 2 and cumulative[seg + 1] < target:
            seg += 1
        seg_len = cumulative[seg + 1] - cumulative[seg]
        t = ((target - cumulative[seg]) / seg_len) if seg_len > 1e-9 else 0.0
        p1, p2 = deduped[seg], deduped[min(seg + 1, len(deduped) - 1)]
        samples.append((float(p1[0]) + t * (float(p2[0]) - float(p1[0])), float(p1[1]) + t * (float(p2[1]) - float(p1[1]))))
    return samples

def build_ideal_template_path(direction, num_samples=100):
    left_points = build_left_first_ideal_waypoints()
    if direction == "right":
        left_points = [(1.0 - x, y) for x, y in left_points]
    return resample_polyline(left_points, num_samples=num_samples)


# ─── Warped Actual Path ───────────────────────────────────────────────────────

def get_closest_horse_point_to_barrel(frame_metrics, barrel_name, arena_diagonal):
    dist_key = f"dist_to_{barrel_name}_px"
    radius = arena_diagonal * BARREL_NEAR_RADIUS_FRACTION
    candidates = [m for m in frame_metrics if m["horse_detected"] and m["horse_center"] and m.get(dist_key) is not None and m[dist_key] <= radius]
    if not candidates:
        return None
    best = min(candidates, key=lambda m: m[dist_key])
    return (float(best["horse_center"][0]), float(best["horse_center"][1]))

def build_warped_actual_path(frame_metrics, identified_barrels, barrel_geometry, direction, original_width, original_height):
    if not all([identified_barrels, barrel_geometry]):
        return []
    top = barrel_geometry.get("top")
    lower_left = barrel_geometry.get("lower_left")
    lower_right = barrel_geometry.get("lower_right")
    if not all([top, lower_left, lower_right]):
        return []
    b1 = identified_barrels.get("barrel1")
    b2 = identified_barrels.get("barrel2")
    b3 = identified_barrels.get("barrel3")
    if not all([b1, b2, b3]):
        return []

    arena_diagonal = math.hypot(original_width, original_height)
    left_x = float(lower_left["center_x"])
    right_x = float(lower_right["center_x"])
    top_y = float(top["center_y"])
    home_y_px = top_y + (original_height - top_y) * 0.85

    if abs(right_x - left_x) < 1.0:
        return []

    def px_to_norm(px_point):
        x, y = px_point
        nx = CANONICAL_LEFT_BARREL_X + ((float(x) - left_x) / (right_x - left_x)) * (CANONICAL_RIGHT_BARREL_X - CANONICAL_LEFT_BARREL_X)
        ny = CANONICAL_TOP_BARREL_Y + ((float(y) - top_y) / max(home_y_px - top_y, 1e-9)) * (CANONICAL_HOME_Y - CANONICAL_TOP_BARREL_Y)
        return (clamp(float(nx), 0.02, 0.98), clamp(float(ny), 0.02, 0.98))

    ideal_path = build_ideal_template_path(direction, num_samples=80)
    ideal_norms = {
        "barrel1": (0.78, 0.50) if direction == "right" else (0.22, 0.50),
        "barrel2": (0.22, 0.50) if direction == "right" else (0.78, 0.50),
        "barrel3": (0.50, 0.17),
    }

    offsets = {}
    for name in ("barrel1", "barrel2", "barrel3"):
        horse_px = get_closest_horse_point_to_barrel(frame_metrics, name, arena_diagonal)
        if horse_px:
            hn = px_to_norm(horse_px)
            offsets[name] = (hn[0] - ideal_norms[name][0], hn[1] - ideal_norms[name][1])
        else:
            offsets[name] = (0.0, 0.0)

    WARP_RADIUS = 0.25
    MAX_WARP = 0.20
    warped = []
    for pt in ideal_path:
        x, y = float(pt[0]), float(pt[1])
        ox, oy = 0.0, 0.0
        for name, ideal_n in ideal_norms.items():
            d = distance((x, y), ideal_n)
            influence = max(0.0, 1.0 - (d / WARP_RADIUS)) if d < WARP_RADIUS else 0.0
            if influence > 0:
                ox += influence * offsets[name][0]
                oy += influence * offsets[name][1]
        warped.append([round(clamp(x + clamp(ox, -MAX_WARP, MAX_WARP), 0.02, 0.98), 4), round(clamp(y + clamp(oy, -MAX_WARP, MAX_WARP), 0.02, 0.98), 4)])
    return warped


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

    output_dir = f"{video_path}_frames"
    os.makedirs(output_dir, exist_ok=True)

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
            image_path = None
            overlay_image_path = None
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

                # Detect barrels every frame now (improved from every other frame)
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

                frame_file = os.path.join(output_dir, f"frame_{idx:03d}.jpg")
                if safe_imwrite(frame_file, frame):
                    image_path = frame_file
                overlay = draw_overlay(frame, horse_detection, barrel_detections, int(target_frame), timestamp_seconds)
                overlay_file = os.path.join(output_dir, f"frame_{idx:03d}_overlay.jpg")
                if safe_imwrite(overlay_file, overlay):
                    overlay_image_path = overlay_file

            sampled_frames.append({
                "percent": round(percent, 4), "frame_index": int(target_frame),
                "timestamp_seconds": timestamp_seconds, "read_success": frame is not None,
                "image_path": image_path, "overlay_image_path": overlay_image_path,
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
                image_path = None
                overlay_image_path = None

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

                    frame_file = os.path.join(output_dir, f"frame_dense_{target_frame:06d}.jpg")
                    if safe_imwrite(frame_file, frame):
                        image_path = frame_file
                    overlay = draw_overlay(frame, horse_detection, barrel_detections, int(target_frame), timestamp_seconds)
                    overlay_file = os.path.join(output_dir, f"frame_dense_{target_frame:06d}_overlay.jpg")
                    if safe_imwrite(overlay_file, overlay):
                        overlay_image_path = overlay_file

                sampled_frames.append({
                    "percent": round(percent, 4), "frame_index": int(target_frame),
                    "timestamp_seconds": timestamp_seconds, "read_success": frame is not None,
                    "image_path": image_path, "overlay_image_path": overlay_image_path,
                    "horse_detection": horse_detection, "barrel_detections": barrel_detections,
                    "barrel_detection_count": len(barrel_detections), "rejection_reason": None, "dense_pass": True,
                })

            sampled_frames.sort(key=lambda f: f["frame_index"])
            all_barrel_detections.sort(key=lambda e: e["frame_index"])

        # ── Final pass with all data ───────────────────────────────────────────
        final_barrels, final_geometry, barrel_id_method = identify_barrels(all_barrel_detections, original_width, original_height)
        final_barrels = remap_barrel_keyed_dict(final_barrels, a2p)
        final_metrics_raw = remap_frame_metric_labels(build_frame_metrics(sampled_frames, final_barrels), a2p)
        
        # ── NEW: Enrich metrics with speed data ───────────────────────────────
        final_metrics = calculate_speed_profile(final_metrics_raw, fps)
        
        final_turns = enforce_turn_order(build_turns(final_metrics))
        splits = build_splits(final_turns, final_metrics, total_run_time_seconds)
        
        # ── NEW: Comprehensive barrel metrics ─────────────────────────────────
        barrel_metrics = build_comprehensive_barrel_metrics(
            final_metrics, final_barrels, final_turns, all_barrel_detections, fps
        )
        
        # ── NEW: Speed summary ────────────────────────────────────────────────
        speed_summary = build_speed_summary(final_metrics, final_turns, fps)
        
        highlights = build_highlights(final_metrics, barrel_metrics)

        # ── Trajectory ────────────────────────────────────────────────────────
        all_track_points = [tuple(f["horse_detection"]["tracking_point"]) if f.get("horse_detection") else None for f in sampled_frames]
        interpolated = interpolate_gaps(all_track_points, INTERPOLATION_MAX_GAP)
        smoothed_points = exponential_smooth([p for p in interpolated if p is not None], SMOOTHING_ALPHA)
        smoothed_points = dedupe_points(smoothed_points, min_dist=6.0)

        # ── Paths ─────────────────────────────────────────────────────────────
        direction = direction_info.get("pattern_direction", "left")
        ideal_template_path = build_ideal_template_path(direction, num_samples=80)
        warped_actual_path = build_warped_actual_path(final_metrics, final_barrels, final_geometry, direction, original_width, original_height)
        if not warped_actual_path:
            warped_actual_path = [[round(p[0], 4), round(p[1], 4)] for p in ideal_template_path]

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
        insights = build_insights(tracking_quality, barrel_detection_summary, direction_info, splits, highlights, barrel_metrics, speed_summary)

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
            "normalized_smoothed_path_points": warped_actual_path,
            "path_map_path": None,
            "tracking_quality": tracking_quality,
            "barrel_detection_summary": barrel_detection_summary,
            "barrel_geometry": final_geometry,
            "pattern_direction": direction,
            "pattern_direction_info": direction_info,
            "identified_barrels": final_barrels,
            "turns": final_turns,
            "splits": splits,
            "barrel_metrics": barrel_metrics,           # NEW: rich per-barrel coaching data
            "speed_summary": speed_summary,             # NEW: run-wide speed analysis
            "normalized_actual_template_path": warped_actual_path,
            "ideal_template_path": [[round(p[0], 4), round(p[1], 4)] for p in ideal_template_path],
            "template_path_comparison": None,
            "speed_scores": {"barrel1": None, "barrel2": None, "barrel3": None},
            "highlights": highlights,
            "frame_metrics": final_metrics,
            "motion_samples": [],
            "sampled_frames": sampled_frames,
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
