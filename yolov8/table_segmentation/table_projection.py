from __future__ import annotations

import cv2
import numpy as np
import ultralytics

from yolov8.detection_results import DetectionResults


class TableProjection:
    def __init__(self, table_corners: list[list[int, int]], table_dimensions: list[int]):
        self.table_corners = self.__reorder_corners(table_corners)
        # Define the destination points of the rectangle (4 corners)
        # Aspect ratio of 8-ball, 9-ball and snooker table is 2:1
        self.dst_width = table_dimensions[0]
        self.dst_height = table_dimensions[1]
        self.dst_points = np.array(
            [[0, 0], [self.dst_width, 0], [self.dst_width, self.dst_height], [0, self.dst_height]], dtype=np.float32
        )
        self.projection_matrix = cv2.getPerspectiveTransform(self.table_corners, self.dst_points)

    def __call__(self, original_image):
        return self.table_projection(original_image)

    def table_projection(self, original_image):
        # Apply the perspective transformation
        return cv2.warpPerspective(original_image.copy(), self.projection_matrix, (self.dst_width, self.dst_height))

    def bounding_box_projection(self, bounding_box: list[float]) -> tuple[float]:
        x, y, w, h = bounding_box
        # Create a list of the 4 corners of the bounding box
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        # Apply the perspective transformation to the bounding box
        projected_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), self.projection_matrix)
        # Get the bounding box of the projected corners
        x, y, w, h = cv2.boundingRect(projected_corners)
        return x, y, w, h
        # # Calculate the bounding box of the projected corners
        # min_x = np.min(projected_corners[:, :, 0])
        # min_y = np.min(projected_corners[:, :, 1])
        # max_x = np.max(projected_corners[:, :, 0])
        # max_y = np.max(projected_corners[:, :, 1])
        # new_x = min_x
        # new_y = min_y
        # new_w = max_x - min_x
        # new_h = max_y - min_y
        # return new_x, new_y, new_w, new_h

    def detection_projection(self, detections: DetectionResults):
        for ball in detections:
            # Update the bounding box of the detection
            new_x, new_y, new_w, new_h = self.bounding_box_projection([ball["x"], ball["y"], ball["w"], ball["h"]])
            ball["x"] = new_x + new_w / 2
            ball["y"] = new_y
            ball["w"] = new_w
            ball["h"] = new_h

    @staticmethod
    def __reorder_corners(table_corners: list[list[int, int]]) -> list[np.float32]:
        for i in range(len(table_corners) - 1):
            # Assume corners are in order (TL, TR, BR, BL)
            # Lowest (TL), Highest (BR), Highest x and low y (TR), Highest y and low x (BL)
            # Reorder if needed
            for j in range(i + 1, len(table_corners)):
                if i == 0:  # TL
                    # if the sum coords is greater and the x coord in smaller
                    if (
                        sum(table_corners[i])
                        > sum(table_corners[j])
                        # and approx_table_corners[i][0] > approx_table_corners[j][0]
                    ):
                        tmp = table_corners[i].copy()
                        table_corners[i] = table_corners[j]
                        table_corners[j] = tmp
                elif i == 1:  # TR
                    # find the corner that has the lowest y coord out of TR, BL, BR
                    if table_corners[i][1] > table_corners[j][1]:
                        tmp = table_corners[i].copy()
                        table_corners[i] = table_corners[j]
                        table_corners[j] = tmp
                elif i == 2:
                    # find the corner that has the greatest x coord out of BL, BR
                    if table_corners[i][0] < table_corners[j][0]:
                        tmp = table_corners[i].copy()
                        table_corners[i] = table_corners[j]
                        table_corners[j] = tmp
        return np.float32(table_corners)
