from __future__ import annotations

import cv2


def record_user_clicks_from_first_frame(
    cap: cv2.VideoCapture, window_name: str, number_of_clicks: int
) -> list[list[int, int]]:
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

    return record_user_clicks_from_image(first_frame, window_name, number_of_clicks)


def record_user_clicks_from_image(image, window_name: str, number_of_clicks: int) -> list[list[int, int]]:
    point_coords = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at ({x}, {y})")
            point_coords.append([x, y])

    # Create a window and display the first frame
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)

    # Set the mouse callback function
    cv2.setMouseCallback(window_name, click_event)

    # Wait for the user to click six times
    while len(point_coords) < number_of_clicks:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release the window
    cv2.destroyWindow(window_name)

    return point_coords


def get_user_pockets(cap: cv2.VideoCapture) -> list[list[int, int]]:
    """
    Get the user to select the pockets on the table by clicking on them.

    :param cap: VideoCapture object to read the video from.
    :return: List of six pocket coordinates.
    """
    return record_user_clicks_from_first_frame(cap, "Select Pockets", 6)


def get_user_corners(cap: cv2.VideoCapture) -> list[list[int, int]]:
    """
    Get the user to select the corners of the table by clicking on them.

    :param cap: VideoCapture object to read the video from.
    :return: List of four corner coordinates.
    """
    return record_user_clicks_from_first_frame(cap, "Select Corners", 4)
