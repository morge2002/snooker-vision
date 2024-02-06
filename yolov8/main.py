import cv2
from ultralytics import YOLO
from PIL import Image

# Config
image_inference = False
video_inference = True
export_model = False
train_model = False

dataset_path = "dataset/snooker-vision.v2i.yolov8/data.yaml"
model_path = "runs/detect/train13/weights/best_openvino_model"
weights_path = "runs/detect/train13/weights/best.pt"

video_path = "test_data/english_pool_video.mp4"

# Train the model
if train_model:
    model = YOLO("yolov8n.pt")
    model.train(data=dataset_path, epochs=100, imgsz=640, batch=-1)

# Export the model
if export_model:
    model = YOLO(weights_path)

    model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# Load trained model
model = YOLO(model_path)

# Inference - Image
if image_inference:
    results = model("test_data/snooker_photo.png")

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        # im.save('results.jpg')  # save image

# Inference - Video
if video_inference:
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            # results = model.predict(frame, device="cpu", max_det=17)

            # Image tracking
            results = model.track(
                frame,
                persist=True,
                tracker="track/custom_bytetrack.yaml",
                device="cpu",
                max_det=17,
            )

            # Visualize the results on the frame
            annotated_frame = results[0].plot(labels=True, conf=False)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

# model(video_path, save=True)
# model.track(
#     video_path,
#     persist=True,
#     tracker="track/custom_bytetrack.yaml",
#     device="cpu",
#     max_det=17,
#     save=True,
# )
