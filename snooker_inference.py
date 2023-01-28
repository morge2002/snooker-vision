import torch

model = torch.hub.load(
    "object_detection_model/YOLOv5/yolov5",
    "custom",
    path="object_detection_model/YOLOv5/yolov5/runs/train/exp3/weights/best.pt",
    source="local",
)
# Image
img = "snooker_photo.png"
# Inference
results = model(img)
# Results, change the flowing to: results.show()
results.show()  # or .show(), .save(), .crop(), .pandas(), etc
