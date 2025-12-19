
from ultralytics import YOLO

# Load the model
model = YOLO('yolov8s-cls.pt')

# Train the model
results = model.train(
    data='../dataset',  # Assuming dataset is in the folder above
    epochs=50,
    imgsz=224,
    batch=16,
    patience=10
)
