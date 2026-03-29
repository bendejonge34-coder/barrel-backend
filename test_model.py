from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/runs_detect/barrel_detector_local/weights/best.pt")

# test image (pick one from your dataset)
image_path = "barrel-ai-dataset/images/frame-001.jpg"

results = model(image_path)

# show result
annotated = results[0].plot()
cv2.imshow("Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()