import json
import sys
import cv2
import os
import math
import contextlib
import io
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

HORSE_CONFIDENCE_THRESHOLD = 0.15
BARREL_CONFIDENCE_THRESHOLD = 0.50
TARGET_SAMPLE_FPS = 5.0
MAX_SAMPLED_FRAMES = 90
MIN_SAMPLED_FRAMES = 30
SMOOTHING_ALPHA = 0.35


def emit_json(payload):
    print(json.dumps(payload))


def fail(error_message, extra=None):
    payload = {
        "ok": False,
        "error": error_message,
    }
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
        return bool(cv2.imwrite(output_path, image))
    except Exception:
        return False


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with contextlib.redirect_stdout(io.StringIO()):
        return YOLO(model_path)


def detect_barrels_in_frame(frame, barrel_model, confidence_threshold=BARREL_CONFIDENCE_THRESHOLD):
    with contextlib.redirect_stdout(io.StringIO()):
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
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "confidence": round(conf, 3),
            "center_x": int((x1 + x2) / 2),
            "center_y": int((y1 + y2) / 2),
        })

    return barrels


def build_horse_candidates(result):
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
        cx, cy = compute_bottom_center(box)
        area = max(0.0, (x2 - x1) * (y2 - y1))

        candidates.append({
            "confidence": round(float(conf), 4),
            "bbox": [
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2), 2),
                round(float(y2), 2),
            ],
            "center": [round(float(cx), 2), round(float(cy), 2)],
            "center_float": (float(cx), float(cy)),
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
        dist = distance(prev_point, candidate["center_float"])
        normalized_dist = dist / diagonal
        score = (candidate["confidence"] * 1.5) - (normalized_dist * 2.0)

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


def adaptive_max_jump(width, height):
    diagonal = math.hypot(width, height) if width > 0 and height > 0 else 1000.0
    return clamp(diagonal * 0.09, 90.0, 260.0)


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


def draw_overlay(frame, horse_detection, trail_points, barrel_detections=None):
    overlay = frame.copy()

    if barrel_detections:
        for barrel in barrel_detections:
            x1 = int(barrel["x1"])
            y1 = int(barrel["y1"])
            x2 = int(barrel["x2"])
            y2 = int(barrel["y2"])
            cx = int(barrel["center_x"])
            cy = int(barrel["center_y"])
            conf = float(barrel["confidence"])

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.circle(overlay, (cx, cy), 6, (255, 255, 0), -1)
            cv2.putText(
                overlay,
                f"Barrel {conf:.2f}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

    if horse_detection is not None:
        x1, y1, x2, y2 = horse_detection["bbox"]
        cx, cy = horse_detection["center"]

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = int(cx), int(cy)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(overlay, (cx, cy), 8, (255, 0, 255), -1)

        cv2.putText(
            overlay,
            f"Horse {horse_detection['confidence']:.2f}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for i in range(1, len(trail_points)):
        p1 = (int(trail_points[i - 1][0]), int(trail_points[i - 1][1]))
        p2 = (int(trail_points[i][0]), int(trail_points[i][1]))
        cv2.line(overlay, p1, p2, (255, 255, 0), 3)

    return overlay


def save_path_map(width, height, raw_points, accepted_points, smooth_points_list, out_path, identified_barrels=None):
    canvas = 255 * (cv2.UMat(height, width, cv2.CV_8UC3).get() * 0 + 1)

    for pt in raw_points:
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 3, (180, 180, 180), -1)

    for pt in accepted_points:
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (0, 165, 255), -1)

    for i in range(1, len(smooth_points_list)):
        p1 = (int(smooth_points_list[i - 1][0]), int(smooth_points_list[i - 1][1]))
        p2 = (int(smooth_points_list[i][0]), int(smooth_points_list[i][1]))
        cv2.line(canvas, p1, p2, (255, 0, 0), 4)

    for pt in smooth_points_list:
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

    if identified_barrels:
        for barrel_name, barrel_info in identified_barrels.items():
            if barrel_info is None:
                continue
            x = int(barrel_info["center_x"])
            y = int(barrel_info["center_y"])
            cv2.circle(canvas, (x, y), 16, (0, 255, 255), 3)
            cv2.putText(
                canvas,
                barrel_name.upper(),
                (x + 8, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 120, 255),
                2,
                cv2.LINE_AA,
            )

    safe_imwrite(out_path, canvas)


def build_sample_frame_indices(frame_count, fps):
    if frame_count <= 0:
        return []

    if fps <= 0:
        fps = 30.0

    step = max(1, int(round(fps / TARGET_SAMPLE_FPS)))
    indices = list(range(0, frame_count, step))

    if indices and indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    elif not indices:
        indices = [0]

    if len(indices) > MAX_SAMPLED_FRAMES:
        stride = len(indices) / float(MAX_SAMPLED_FRAMES)
        reduced = []
        cursor = 0.0
        while int(cursor) < len(indices):
            reduced.append(indices[int(cursor)])
            cursor += stride
        indices = reduced
        if indices[-1] != frame_count - 1:
            indices.append(frame_count - 1)

    if len(indices) < MIN_SAMPLED_FRAMES and frame_count > MIN_SAMPLED_FRAMES:
        desired = min(MIN_SAMPLED_FRAMES, frame_count)
        indices = sorted(
            set(int(i * (frame_count - 1) / max(desired - 1, 1)) for i in range(desired))
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


def cluster_barrel_points(points, width, height, max_clusters=3, iterations=12):
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
            "source_points": assigned_points,
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
            matched["source_points"].extend(cluster["source_points"])

    merged = sorted(merged, key=lambda c: c["detection_count"], reverse=True)[:3]
    merged = sorted(merged, key=lambda c: c["center_x"])

    return merged


def identify_barrels(all_barrel_detections, width, height):
    points = flatten_barrel_points(all_barrel_detections)
    clusters = cluster_barrel_points(points, width, height, max_clusters=3, iterations=12)

    identified = {
        "barrel1": None,
        "barrel2": None,
        "barrel3": None,
    }

    for idx, cluster in enumerate(clusters):
        if idx >= 3:
            break

        label = BARREL_LABELS[idx]
        identified[label] = {
            "center_x": round(float(cluster["center_x"]), 2),
            "center_y": round(float(cluster["center_y"]), 2),
            "detection_count": int(cluster["detection_count"]),
            "average_confidence": round(float(cluster["average_confidence"]), 3),
        }

    return identified


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
                float(horse_detection["center"][0]),
                float(horse_detection["center"][1]),
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
    threshold = max(apex_distance * 1.8, apex_distance + 40.0, diagonal * 0.08)

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


def average_values(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def compute_path_length(points):
    if len(points) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(points)):
        total += distance(points[i - 1], points[i])

    return total


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

        barrel_center = (barrel_info["center_x"], barrel_info["center_y"])

        entry_speeds = [s["speed_px_per_sec"] for s in turn_samples[:2]]
        exit_speeds = [s["speed_px_per_sec"] for s in turn_samples[-2:]]

        path_points = [tuple(s["horse_center"]) for s in turn_samples if s["horse_center"] is not None]
        path_length_px = compute_path_length(path_points)

        distance_key = f"dist_to_{barrel_name}_px"
        turn_distances = [s[distance_key] for s in turn_samples if s.get(distance_key) is not None]

        heading_start = start_sample["heading_deg"] if start_sample else None
        heading_end = end_sample["heading_deg"] if end_sample else None
        heading_change = angle_difference_deg(heading_start, heading_end)

        entry_angle_deg = None
        exit_angle_deg = None

        if start_sample and apex_sample:
            entry_angle_deg = angle_degrees(start_sample["horse_center"], apex_sample["horse_center"])

        if apex_sample and end_sample:
            exit_angle_deg = angle_degrees(apex_sample["horse_center"], end_sample["horse_center"])

        metrics[barrel_name] = {
            "start_frame": int(turn["start_frame"]),
            "apex_frame": int(turn["apex_frame"]),
            "end_frame": int(turn["end_frame"]),
            "start_timestamp_seconds": turn["start_timestamp_seconds"],
            "apex_timestamp_seconds": turn["apex_timestamp_seconds"],
            "end_timestamp_seconds": turn["end_timestamp_seconds"],
            "min_distance_px": round_or_none(turn["min_distance_px"], 2),
            "average_turn_distance_px": round_or_none(average_values(turn_distances), 2),
            "entry_speed_px_per_sec": round_or_none(average_values(entry_speeds), 2),
            "exit_speed_px_per_sec": round_or_none(average_values(exit_speeds), 2),
            "path_length_px": round_or_none(path_length_px, 2),
            "heading_change_deg": round_or_none(heading_change, 2),
            "entry_angle_deg": round_or_none(entry_angle_deg, 2),
            "exit_angle_deg": round_or_none(exit_angle_deg, 2),
            "turn_sample_count": len(turn_samples),
            "barrel_center": round_point(barrel_center, 2),
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


def choose_best_barrel(barrel_metrics):
    candidates = []

    for barrel_name, metric in barrel_metrics.items():
        if metric is None:
            continue

        score = 0.0
        avg_turn_distance = metric.get("average_turn_distance_px")
        heading_change = metric.get("heading_change_deg")
        exit_speed = metric.get("exit_speed_px_per_sec")

        if avg_turn_distance is not None:
            score -= avg_turn_distance * 0.25
        if heading_change is not None:
            score += heading_change * 0.35
        if exit_speed is not None:
            score += exit_speed * 0.02

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

        if path_length is not None:
            score -= path_length * 0.02
        if min_distance is not None:
            score -= min_distance * 0.2
        if heading_change is not None:
            score += heading_change * 0.3

        candidates.append((barrel_name, score))

    if not candidates:
        return None

    return max(candidates, key=lambda x: x[1])[0]


def build_insights(barrel_metrics, splits, identified_barrels):
    insights = []

    detected_barrel_count = sum(1 for v in identified_barrels.values() if v is not None)
    if detected_barrel_count < 3:
        insights.append(
            "Barrel identification is incomplete. The detector found fewer than three stable barrel positions, so turn analysis is limited."
        )

    for barrel_name in BARREL_LABELS:
        metric = barrel_metrics.get(barrel_name)
        if metric is None:
            continue

        min_distance = metric.get("min_distance_px")
        entry_speed = metric.get("entry_speed_px_per_sec")
        exit_speed = metric.get("exit_speed_px_per_sec")
        heading_change = metric.get("heading_change_deg")
        avg_turn_distance = metric.get("average_turn_distance_px")

        label = barrel_name.upper()

        if min_distance is not None and min_distance < 45:
            insights.append(
                f"{label}: you likely crowded the barrel. Minimum distance was only {min_distance:.1f}px, which suggests a very tight pocket."
            )
        elif min_distance is not None and min_distance > 120:
            insights.append(
                f"{label}: the turn appears wide. Minimum distance was {min_distance:.1f}px, which usually means extra path length around the barrel."
            )

        if entry_speed is not None and exit_speed is not None:
            speed_delta = exit_speed - entry_speed
            if speed_delta < -25:
                insights.append(
                    f"{label}: exit speed dropped versus entry speed. That usually means momentum was lost through the turn."
                )
            elif speed_delta > 15:
                insights.append(
                    f"{label}: strong acceleration out of the turn. Exit speed improved after the apex."
                )

        if heading_change is not None and heading_change < 35:
            insights.append(
                f"{label}: heading change through the turn was limited, which can indicate a flatter arc instead of a committed wrap."
            )
        elif heading_change is not None and heading_change > 95:
            insights.append(
                f"{label}: heading change was aggressive. That can indicate a strong wrap, but only if the exit line stays clean."
            )

        if avg_turn_distance is not None and avg_turn_distance > 140:
            insights.append(
                f"{label}: average turn distance stayed high, which points to a larger-than-ideal turn radius."
            )

    if splits.get("start_to_barrel1_seconds") is not None:
        insights.append(
            f"Start to Barrel 1 estimated split: {splits['start_to_barrel1_seconds']:.3f}s."
        )
    if splits.get("barrel1_to_barrel2_seconds") is not None:
        insights.append(
            f"Barrel 1 to Barrel 2 estimated split: {splits['barrel1_to_barrel2_seconds']:.3f}s."
        )
    if splits.get("barrel2_to_barrel3_seconds") is not None:
        insights.append(
            f"Barrel 2 to Barrel 3 estimated split: {splits['barrel2_to_barrel3_seconds']:.3f}s."
        )
    if splits.get("barrel3_to_home_seconds") is not None:
        insights.append(
            f"Barrel 3 to home estimated split: {splits['barrel3_to_home_seconds']:.3f}s."
        )

    best_barrel = choose_best_barrel(barrel_metrics)
    if best_barrel is not None:
        insights.append(
            f"Best barrel estimate: {best_barrel.upper()}. This barrel combined tighter geometry with stronger exit characteristics than the others."
        )

    best_turn = choose_best_turn(barrel_metrics)
    if best_turn is not None:
        insights.append(
            f"Best turn estimate: {best_turn.upper()}. This turn appears to balance path efficiency, barrel proximity, and directional change better than the others."
        )

    return insights[:12]


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

    try:
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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0

        if frame_count <= 0 or width <= 0 or height <= 0:
            fail(
                "Video metadata could not be read correctly.",
                {
                    "video_path": video_path,
                    "frame_count": frame_count,
                    "fps": fps,
                    "width": width,
                    "height": height,
                },
            )
            return

        sample_indices = build_sample_frame_indices(frame_count, fps)
        max_jump = adaptive_max_jump(width, height)

        sampled_frames = []
        all_barrel_detections = []

        horse_detected_count = 0
        read_success_count = 0
        rejected_jump_count = 0
        missed_detection_count = 0

        raw_trajectory_points = []
        accepted_points = []
        smoothed_points = []

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

            if ret and frame is not None:
                read_success_count += 1

                image_path = f"{video_path}_frame_{idx:03d}.jpg"
                safe_imwrite(image_path, frame)

                barrel_detections = detect_barrels_in_frame(
                    frame,
                    barrel_model,
                    confidence_threshold=BARREL_CONFIDENCE_THRESHOLD,
                )

                all_barrel_detections.append({
                    "frame_index": int(target_frame),
                    "timestamp_seconds": timestamp_seconds,
                    "barrels": barrel_detections,
                })

                with contextlib.redirect_stdout(io.StringIO()):
                    horse_results = horse_model.predict(
                        source=frame,
                        conf=HORSE_CONFIDENCE_THRESHOLD,
                        classes=[HORSE_CLASS_ID],
                        verbose=False,
                    )

                candidates = []
                if horse_results and len(horse_results) > 0:
                    candidates = build_horse_candidates(horse_results[0])

                best_candidate = choose_best_candidate(
                    candidates,
                    previous_accepted_point,
                    width,
                    height,
                )

                if best_candidate is not None:
                    horse_detected_count += 1
                    horse_detection = {
                        "confidence": best_candidate["confidence"],
                        "bbox": best_candidate["bbox"],
                        "center": best_candidate["center"],
                    }

                    current_point = best_candidate["center_float"]
                    raw_trajectory_points.append(current_point)

                    if previous_accepted_point is None:
                        accepted_points.append(current_point)
                        previous_accepted_point = current_point
                    else:
                        jump_distance = distance(previous_accepted_point, current_point)

                        if jump_distance <= max_jump:
                            accepted_points.append(current_point)
                            previous_accepted_point = current_point
                        else:
                            rejected_jump_count += 1
                            rejection_reason = f"jump_rejected_{round(jump_distance, 2)}px"
                else:
                    missed_detection_count += 1

                smoothed_points = smooth_points(accepted_points, alpha=SMOOTHING_ALPHA)

                overlay = draw_overlay(frame, horse_detection, smoothed_points, barrel_detections)
                overlay_image_path = f"{video_path}_frame_{idx:03d}_overlay.jpg"
                safe_imwrite(overlay_image_path, overlay)

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

        barrel_detection_summary = summarize_barrel_detections(all_barrel_detections)
        identified_barrels = identify_barrels(all_barrel_detections, width, height)

        frame_metrics = build_frame_metrics(sampled_frames, identified_barrels)
        motion_samples = build_motion_samples(frame_metrics)
        turns = build_turns(frame_metrics, width, height)
        barrel_metrics = build_barrel_metrics(turns, motion_samples, identified_barrels)
        splits = build_splits(turns, motion_samples)
        insights = build_insights(barrel_metrics, splits, identified_barrels)

        path_map_path = None
        if width > 0 and height > 0 and len(smoothed_points) > 1:
            path_map_path = f"{video_path}_path_map.jpg"
            save_path_map(
                width,
                height,
                raw_trajectory_points,
                accepted_points,
                smoothed_points,
                path_map_path,
                identified_barrels=identified_barrels,
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
        }

        smoothed_path_points = [round_point(pt, 2) for pt in smoothed_points]
        normalized_smoothed_path_points = normalize_points_to_unit_box(smoothed_points)

        output = {
            "ok": True,
            "message": "Video opened and barrel-aware run analysis was completed.",
            "video_path": video_path,
            "frame_count": frame_count,
            "fps": round(fps, 3),
            "width": width,
            "height": height,
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
            "identified_barrels": identified_barrels,

            "turns": turns,
            "splits": splits,
            "barrel_metrics": barrel_metrics,
            "insights": insights,

            "frame_metrics": frame_metrics,
            "motion_samples": motion_samples,
            "sampled_frames": sampled_frames,
        }

        emit_json(output)

    except Exception as runtime_error:
        fail(
            "Python analysis crashed.",
            {
                "details": str(runtime_error),
                "video_path": video_path,
            },
        )
    finally:
        cap.release()


if __name__ == "__main__":
    main()