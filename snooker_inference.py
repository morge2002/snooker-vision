import torch
import cv2

classes = {0: "ball", 1: "table"}

model = torch.hub.load(
    "object_detection_model/YOLOv5/yolov5",
    "custom",
    path="object_detection_model/YOLOv5/yolov5/runs/train/exp5/weights/best.pt",
    source="local",
)
# Image
img = "snooker_photo.png"
video = "snooker_video.mp4"
# Inference
results = model(img)

cap = cv2.VideoCapture(video)

while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    print(f"classes: {model.classes}")
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    print(f"labels: {labels}, coords: {coords}")
    # Results
    # results.show()  # or .show(), .save(), .crop(), .pandas(), etc

cap.release()
cv2.destroyAllWindows()

from deep_sort_realtime.deepsort_tracker import DeepSort

# All default values
object_tracker = DeepSort(
    max_age=5,  # Number of frames that can be missed before the tracking is discarded
    n_init=2,  # Number of consecutive detections before the tracking is started for an object
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=None,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon=False,
    today=None,
)
