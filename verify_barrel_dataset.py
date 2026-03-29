from pathlib import Path

DATASET_ROOT = Path("datasets/barrel_detector")
TRAIN_IMAGES = DATASET_ROOT / "images" / "train"
VAL_IMAGES = DATASET_ROOT / "images" / "val"
TRAIN_LABELS = DATASET_ROOT / "labels" / "train"
VAL_LABELS = DATASET_ROOT / "labels" / "val"

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS])

def get_label_path(image_path: Path, labels_folder: Path):
    return labels_folder / f"{image_path.stem}.txt"

def validate_label_file(label_path: Path):
    errors = []
    if not label_path.exists():
        errors.append("Missing label file")
        return errors

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        errors.append("Empty label file")
        return errors

    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"Line {i}: Expected 5 values, found {len(parts)}")
            continue

        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            errors.append(f"Line {i}: Non-numeric value found")
            continue

        if class_id != 0:
            errors.append(f"Line {i}: Invalid class_id {class_id}, expected 0")

        for name, value in [
            ("x_center", x_center),
            ("y_center", y_center),
            ("width", width),
            ("height", height),
        ]:
            if not (0.0 <= value <= 1.0):
                errors.append(f"Line {i}: {name} out of range: {value}")

        if width <= 0 or height <= 0:
            errors.append(f"Line {i}: width/height must be > 0")

    return errors

def verify_split(images_folder: Path, labels_folder: Path, split_name: str):
    print(f"\n--- VERIFYING {split_name.upper()} ---")
    print(f"Images folder: {images_folder}")
    print(f"Labels folder: {labels_folder}")

    if not images_folder.exists():
        print("ERROR: Images folder does not exist")
        return False
    if not labels_folder.exists():
        print("ERROR: Labels folder does not exist")
        return False

    images = get_images(images_folder)
    print(f"Found {len(images)} image files")

    if len(images) == 0:
        print("ERROR: No images found")
        return False

    ok = True
    missing_labels = 0
    bad_labels = 0

    for image_path in images:
        label_path = get_label_path(image_path, labels_folder)
        errors = validate_label_file(label_path)

        if "Missing label file" in errors:
            missing_labels += 1
            ok = False
            print(f"[MISSING LABEL] {image_path.name}")
            continue

        if errors:
            bad_labels += 1
            ok = False
            print(f"[BAD LABEL] {image_path.name}")
            for err in errors:
                print(f"  - {err}")

    label_files = sorted(labels_folder.glob("*.txt"))
    image_stems = {p.stem for p in images}
    orphan_labels = [p.name for p in label_files if p.stem not in image_stems]

    if orphan_labels:
        ok = False
        print("\n[ORPHAN LABEL FILES FOUND]")
        for name in orphan_labels:
            print(f"  - {name}")

    print(f"\nSummary for {split_name}:")
    print(f"  Images: {len(images)}")
    print(f"  Missing labels: {missing_labels}")
    print(f"  Bad labels: {bad_labels}")
    print(f"  Orphan labels: {len(orphan_labels)}")

    return ok

def main():
    print("Checking barrel detector dataset structure...")

    train_ok = verify_split(TRAIN_IMAGES, TRAIN_LABELS, "train")
    val_ok = verify_split(VAL_IMAGES, VAL_LABELS, "val")

    overall_ok = train_ok and val_ok

    yaml_path = DATASET_ROOT / "barrel_dataset.yaml"
    print(f"\nYAML file check: {yaml_path}")
    if yaml_path.exists():
        print("YAML file found")
    else:
        print("ERROR: barrel_dataset.yaml not found")
        overall_ok = False

    print("\n=== FINAL RESULT ===")
    if overall_ok:
        print("Dataset looks valid and ready for YOLO training.")
    else:
        print("Dataset has issues. Fix them before training.")

if __name__ == "__main__":
    main()