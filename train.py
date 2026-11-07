from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="Glass defect detection.v1i.yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="glass_defect",
)

print("Training complete!")
print("Best model saved at: runs/detect/glass_defect/weights/best.pt")