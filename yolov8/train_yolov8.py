from ultralytics import YOLO

dataset_path = "dataset/snooker-vision.v2i.yolov8/data.yaml"
weights_path = "runs/detect/train13/weights/best.pt"

# Train the model
model = YOLO("yolov8n.pt")
model.train(data=dataset_path, epochs=100, imgsz=640, batch=-1)

# Export the model
# model = YOLO(weights_path)
# model.export(format="openvino")  # creates 'yolov8n_openvino_model/'
