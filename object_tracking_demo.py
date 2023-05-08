import os
import time
import numpy as np

import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort


class YoloDetector:
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        # print(self.classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device: ", self.device)

    def load_model(self, model_name):
        # if model_name:
        #     model = torch.hub.load(
        #         "ultralytics/yolov5", "custom", path=model_name, force_reload=True
        #     )
        # else:
        #     model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        model = torch.hub.load(
            "object_detection_model/YOLOv5/yolov5",
            "custom",
            path="object_detection_model/YOLOv5/yolov5/runs/train/exp5/weights/best.pt",
            source="local",
        )
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))
        # frame = frame.to(self.device)

        results = self.model(frame)
        print(f"results: ", results)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cord

    def class_to_label(self, x):
        classes = {0: "ball", 1: "table"}
        return classes[int(x)]

    def plot_boxes(self, results, frame, height, width, confidence_threshold=0.3):
        labels, cord = results
        detections = []

        x_shape, y_shape = width, height
        for i in range(len(labels)):
            row = cord[i]

            if row[4] >= confidence_threshold:
                x1, y1, x2, y2 = (
                    int(row[0] * x_shape),
                    int(row[1] * y_shape),
                    int(row[2] * x_shape),
                    int(row[3] * y_shape),
                )

                if self.class_to_label(labels[i]) == "ball":
                    x_center = x1 + (x2 - x1)
                    y_center = y1 + ((y2 - y1) / 2)

                    tlwh = np.asarray(
                        [x1, y1, int(x2 - x1), int(y2 - y1)], dtype=np.float32
                    )
                    confidence = float(row[4].item())
                    feature = "ball"

                    # ([left,top,w,h], confidence, detection_class)
                    detections.append(
                        ([x1, y1, int(x2 - x1), int(y2 - y1)], confidence, feature)
                    )

        return frame, detections


cap = cv2.VideoCapture("snooker_video.mp4")

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = YoloDetector(
    model_name="object_detection_model/YOLOv5/yolov5/runs/train/exp5/weights/best.pt"
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

object_tracker = DeepSort(
    max_age=10,
    n_init=2,
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

while cap.isOpened():
    succes, img = cap.read()

    start = time.perf_counter()

    results = detector.score_frame(img)
    img, detections = detector.plot_boxes(
        results, img, height=img.shape[0], width=img.shape[1], confidence_threshold=0
    )
    print(f"detections: {detections}")
    tracks = object_tracker.update_tracks(
        detections, frame=img
    )  # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
    print(f"tracks: {tracks}")
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()

        bbox = ltrb

        cv2.rectangle(
            img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            img,
            "ID: " + str(track_id),
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    end = time.perf_counter()
    totalTime = end - start
    fps = 1 / totalTime

    cv2.putText(
        img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2
    )
    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()
