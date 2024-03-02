from __future__ import annotations

import cv2


def get_user_pockets(cap: cv2.VideoCapture) -> list[list[int, int]]:
    """
    Get the user to select the pockets on the table by clicking on them.

    :param cap: VideoCapture object to read the video from
    :return: List of six pocket coordinates
    """
    pocket_coords = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at ({x}, {y})")
            pocket_coords.append([x, y])

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
    while len(pocket_coords) < 6:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release the window
    cv2.destroyWindow("Pocket Selection")

    return pocket_coords
