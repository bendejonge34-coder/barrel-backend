from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="barrel-ai-dataset/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        name="barrel_detector_local",
        project="runs_detect",
        device="cpu"
    )

if __name__ == "__main__":
    main()