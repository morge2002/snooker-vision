from __future__ import annotations

import os

import cv2
from ultralytics import YOLO

from yolov8.ball import Balls
from yolov8.detection_results import DetectionResults
from yolov8.pockets import Pockets
from yolov8.pot_detection import PotDetector
from yolov8.table_segmentation import TableProjection
from yolov8.user_input import UserInput


class CueVision:
    # Relative to the current file
    tracker_file_path = os.path.join(os.path.dirname(__file__), "track/custom_bytetrack.yaml")
    pocket_rois = [5, 25, 50]
    table_dimensions = [320, 640]

    def __init__(self, model_path: str, pocket_rois: list[int] = None, table_dimensions: list[int] = None):
        self.model_path = model_path
        self.model = YOLO(model_path)
        if pocket_rois is not None:
            self.pocket_rois = pocket_rois
        if table_dimensions is not None:
            self.table_dimensions = table_dimensions

    def __call__(self, video_path: str, show_video=True) -> dict[int, tuple]:
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        user_input = UserInput(ask_user_to_reselect_corners=False)

        # Get and store the user-selected corners for table projection
        corner_coordinates = user_input.get_user_corners(cap, video_path)
        table_projection = TableProjection(corner_coordinates, self.table_dimensions)

        # Project the first frame of the video to the table
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Unable to read first frame from video")
            exit()
        projected_frame = table_projection(first_frame)

        # Get and store the user-selected pockets for pocket detection
        pocket_coordinates = user_input.get_user_pockets(projected_frame, video_path)

        # Class to track ball state
        balls = Balls()

        # Class to store the pockets and update which balls are in what pocket ROIs
        pockets_tracker = Pockets(balls, pocket_coordinates, self.pocket_rois)

        # Class to detect pots
        pot_detector = PotDetector(balls, pockets_tracker)

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Image detection and tracking
                yolo_results = self.model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker_file_path,
                    max_det=17,
                    verbose=False,
                )
                detection_results: DetectionResults = DetectionResults(yolo_results[0])

                # Skip the frame if no balls are detected/tracked
                if detection_results.is_empty():
                    continue

                # Project the frame to a 2:1 table
                projected_frame = table_projection(frame)
                table_projection.detection_projection(detection_results)

                annotated_frame = frame

                if show_video:
                    # Visualize the detections and tracking
                    annotated_frame = yolo_results[0].plot(labels=True, conf=False)

                    # Draw the pocket coordinates and RIOs on the frame
                    for pocket in pocket_coordinates:
                        for roi in self.pocket_rois:
                            cv2.circle(projected_frame, (pocket[0], pocket[1]), roi, (0, 0, 255), 2)

                    # Draw the ball center points
                    for ball in detection_results:
                        cv2.circle(
                            projected_frame,
                            (int(ball["x"]), int(ball["y"])),
                            2,
                            (255, 0, 255),
                            2,
                        )

                # Update the ball positions and metadata
                balls.update(detection_results)

                # Update the pocket rois for the balls
                pockets_tracker(detection_results)

                # Detect pots
                pot_detector(detection_results, round(cap.get(cv2.CAP_PROP_POS_MSEC)), projected_frame)

                if show_video:
                    # Display the annotated frame
                    cv2.imshow("YOLOv8 Inference", annotated_frame)
                    cv2.imshow("Pot Detection", projected_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
        return pot_detector.successful_pots


if __name__ == "__main__":
    # Path to the YOLOv8 model
    model_path = os.path.join(os.path.dirname(__file__), "./runs/detect/train14/weights/best_openvino_model")

    # Path to the video file
    # video_path = "../tests/test_data/english_pool_dish_one_frame.mp4"
    # video_path = "../tests/test_data/snooker_clearance_portrait.mov"
    # video_path = "../tests/test_data/9_ball_clearance.mp4"
    # video_path = "../tests/test_data/english_pool_clearance_1.mp4"
    video_path = "../tests/test_data/barton_snooker_clearance_10_pots.mp4"
    # Run the CueVision pipeline
    cue_vision = CueVision(model_path, pocket_rois=[5, 25, 50])
    pots = cue_vision(video_path, show_video=True)
    print(pots)
