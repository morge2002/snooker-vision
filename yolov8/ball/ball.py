from __future__ import annotations

import cv2
import numpy as np


class Ball:
    def __init__(self, ball_id: int, x: float, y: float, frame=None, width: float = None, height: float = None) -> None:
        self.ball_id: int = ball_id
        self.x: float = x
        self.y: float = y
        self.__direction_vector: list[float] = [0, 0]
        self.__direction_magnitude: int = 0
        self.__direction_vector_is_current: bool = True
        self.pocketed: bool = False
        self.coordinate_history: list[list[float]] = []
        # Number of frames the ball has been missing since the last sighting
        self.missing_frame_count: int = 0
        self.colour = None
        # Calculate the colour of the ball
        if frame is not None and width is not None and height is not None:
            self.__calculate_colour(frame, width, height)

    # TODO: Should the coordinate history contain the current position?
    def update_position(self, new_x: float, new_y: float) -> None:
        self.coordinate_history.append([self.x, self.y])
        self.x = new_x
        self.y = new_y
        self.__direction_vector_is_current = False
        self.missing_frame_count = 0

    def missed_frame(self) -> None:
        self.missing_frame_count += 1

    def get_direction_vector(self) -> list[float]:
        """
        Calculate the direction vector of the ball's movement.

        The direction of the ball is only calculated on retrieval of the direction vector. This is to avoid unnecessary
        calculations if the direction vector is not needed.

        :return: The direction vector of the ball's movement
        """
        if self.__direction_vector_is_current:
            return self.__direction_vector
        if len(self.coordinate_history) == 0:
            return [0.0, 0.0]

        r1 = np.array(self.coordinate_history[-1])  # Previous position (x1, y1)
        r2 = np.array([self.x, self.y])  # Final position (x2, y2)

        displacement = r2 - r1
        # If the displacement is zero, return the displacement
        # if not displacement.any():
        #     return displacement
        # Apply a threshold to the displacement to avoid noise in the direction vector
        if np.linalg.norm(displacement) < 1:
            self.__direction_vector = [0.0, 0.0]
            self.__direction_magnitude = 0
            self.__direction_vector_is_current = True
            return [0.0, 0.0]

        magnitude = np.linalg.norm(displacement)

        # Normalize displacement vector to obtain direction vector
        direction_vector = displacement / magnitude

        self.__direction_vector = direction_vector
        self.__direction_magnitude = magnitude
        self.__direction_vector_is_current = True
        return direction_vector

    def get_magnitude(self):
        # This function will update the magnitude also
        self.get_direction_vector()
        return self.__direction_magnitude

    # TODO: The cue ball is mainly green since the table around and the green reflections mean that green is the main
    #  colour
    def __calculate_colour(self, frame, width: float, height: float):
        # Convert the image from BGR to HSV
        image_hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
        bounding_box_image = image_hsv[
            int(self.y - height / 2) : int(self.y + height / 2),
            int(self.x - width / 2) : int(self.x + width / 2),
        ]
        # Show the image
        # bounding_box_image_brg = cv2.cvtColor(bounding_box_image.copy(), cv2.COLOR_HSV2BGR)
        # cv2.namedWindow(f"Bounding box ({self.ball_id})", cv2.WINDOW_NORMAL)  # explicit window creation
        # cv2.imshow(f"Bounding box ({self.ball_id})", bounding_box_image_brg)
        # cv2.resizeWindow(f"Bounding box ({self.ball_id})", (100, 100))
        # Reshape the image to a 2D array of pixels
        pixels = bounding_box_image.reshape((-1, 3))
        # Convert to float32
        pixels = np.float32(pixels)

        # Define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3  # Number of clusters is three for the background, shadow and ball colour
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert back to 8 bit values
        centers = np.uint8(centers)

        # Find the most common color
        counts = np.bincount(labels.flatten())
        main_color = centers[np.argmax(counts)]

        # print(f"Main colour rgb {self.ball_id}: {cv2.cvtColor(np.uint8([[main_color]]), cv2.COLOR_HSV2RGB)}")

        self.colour = main_color
        return main_color
