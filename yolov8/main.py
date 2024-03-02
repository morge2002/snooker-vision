import cv2
from ultralytics import YOLO
from PIL import Image

from yolov8.pot_detection import PocketROIHeuristic

# Config
image_inference = False
video_inference = True
export_model = False
train_model = False

dataset_path = "dataset/snooker-vision.v2i.yolov8/data.yaml"
model_path = "runs/detect/train13/weights/best_openvino_model"
weights_path = "runs/detect/train13/weights/best.pt"

video_path = "test_data/english_pool_dish_one_frame.mp4"

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

    # Store pocket coordinates
    pocket_coordinates = []

    def click_event(event, x, y, flags, param):
        global pocket_coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at ({x}, {y})")
            pocket_coordinates.append([x, y])

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file")
        exit()

    # Read the first frame
    ret, first_frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Unable to read first frame from video")
        exit()

    # Create a window and display the first frame
    cv2.namedWindow("Pocket Selection")
    cv2.imshow("Pocket Selection", first_frame)

    # Set the mouse callback function
    cv2.setMouseCallback("Pocket Selection", click_event)

    # Wait for the user to click six times
    while len(pocket_coordinates) < 6:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release the window
    cv2.destroyWindow("Pocket Selection")

    rios = [5, 10, 20, 50]
    pot_detector = PocketROIHeuristic(pocket_coordinates, rios)

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

            # Draw the pocket coordinates and RIOs on the frame
            for pocket in pocket_coordinates:
                for roi in rios:
                    cv2.circle(annotated_frame, (pocket[0], pocket[1]), roi, (0, 0, 255), 2)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Detect pot using ROI heuristic
            pot_detector(results[0])

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
