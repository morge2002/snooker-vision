from __future__ import annotations

import json
import os

import cv2


class UserInput:
    """
    Class to handle user input for selecting the corners and pockets of the table. The coordinates are stored in a JSON
    file for future use.
    """

    table_coordinates_filename = os.path.join(os.path.dirname(__file__), "./table_coordinates.json")

    def __init__(self, ask_user_to_reselect_corners: bool = True):
        # Ask the user to reselect the corners and pockets or use the previously stored coordinates if available
        self.ask_user_to_reselect_corners = ask_user_to_reselect_corners

    def __get_table_coordinates(self) -> dict:
        """
        Get the table coordinates from the JSON file.

        :return: Dictionary containing the table coordinates.
        """
        if not os.path.exists(self.table_coordinates_filename):
            with open(self.table_coordinates_filename, "w") as file:
                json.dump({}, file)
                return {}
        with open(self.table_coordinates_filename, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}

    def __write_table_coordinates(self, table_coords: dict):
        """
        Write the table coordinates to the JSON file.

        :param table_coords: Dictionary containing the table coordinates.
        """
        with open(self.table_coordinates_filename, "w") as file:
            try:
                json.dump(table_coords, file)
            except json.JSONDecodeError:
                print("Error: Unable to write table coordinates to file")

    def record_user_clicks_from_first_frame(
        self, cap: cv2.VideoCapture, window_name: str, number_of_clicks: int
    ) -> list[list[int, int]]:
        """
        Record user clicks from the first frame of the video.

        :param cap: VideoCapture object to read the video from.
        :param window_name: Window name to display the first frame.
        :param number_of_clicks: Number of clicks to record.
        :return: List of click coordinates.
        """
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

        return self.record_user_clicks_from_image(first_frame, window_name, number_of_clicks)

    @staticmethod
    def record_user_clicks_from_image(image, window_name: str, number_of_clicks: int) -> list[list[int, int]]:
        """
        Record user clicks from an image.

        :param image: Image to display and record clicks from.
        :param window_name: Window name to display the image.
        :param number_of_clicks: Number of clicks to record.
        :return: List of click coordinates.
        """
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

    def get_user_pockets(self, first_frame, video_filename: str) -> list[list[int, int]]:
        """
        Get the user to select the pockets on the table by clicking on them.

        :param first_frame: VideoCapture object to read the video from.
        :param video_filename: Path to the video file as identifier in config file.
        :return: List of six pocket coordinates.
        """
        rerecord_corners = "n"
        # Check is coords stored in file in the current directory
        table_coords = self.__get_table_coordinates()
        if (
            video_filename in table_coords
            and "pockets" in table_coords[video_filename]
            and len(table_coords[video_filename]["pockets"]) == 6
        ):
            if self.ask_user_to_reselect_corners:
                # Check if the user wants to use the previously stored coordinates
                rerecord_corners = input("Would you like to re-record the table pocket coords? (y/n): ")
        else:
            print("No pockets found in table_coordinates.json")
            rerecord_corners = "y"

        if rerecord_corners.lower() == "n":
            return table_coords[video_filename]["pockets"]

        pockets = self.record_user_clicks_from_image(first_frame, "Select Pockets", 6)

        # Store coords in file in the current directory
        table_coords[video_filename]["pockets"] = pockets
        self.__write_table_coordinates(table_coords)
        return pockets

    def get_user_corners(self, cap: cv2.VideoCapture, video_filename: str) -> list[list[int, int]]:
        """
        Get the user to select the corners of the table by clicking on them.

        :param cap: VideoCapture object to read the video from.
        :param video_filename: Path to the video file as identifier in config file.
        :return: List of four corner coordinates.
        """
        rerecord_corners = "n"
        table_coords = self.__get_table_coordinates()
        # Check is coords stored in file in the current directory
        if video_filename in table_coords:
            if self.ask_user_to_reselect_corners:
                # Check if the user wants to use the previously stored coordinates
                rerecord_corners = input("Would you like to re-record the table corners? (y/n): ")
        else:
            print("No corners found in table_coordinates.json")
            rerecord_corners = "y"

        if rerecord_corners.lower() == "n":
            return table_coords[video_filename]["corners"]

        corners = self.record_user_clicks_from_first_frame(cap, "Select Corners", 4)

        # Store coords in file in the current directory
        if not os.path.exists(self.table_coordinates_filename):
            self.__write_table_coordinates({video_filename: {"corners": corners, "pockets": []}})
            return corners

        if video_filename not in table_coords:
            table_coords[video_filename] = {"corners": [], "pockets": []}
        table_coords[video_filename]["corners"] = corners

        self.__write_table_coordinates(table_coords)
        return corners
