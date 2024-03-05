from __future__ import annotations

import cv2
import numpy as np
import ultralytics.engine.results


# TODO: Need to add logic to specify if a ball is headed to a pocket. The PotDetector will use this along with other
#  heuristics to determine if a pot has been made
class LinearExtrapolationHeuristic:
    def __init__(self, pocket_coordinates: list[list[int, int]] = None, **kwargs):
        # Data structure to keep track of the balls
        # Structure: {ball_id: {"x": x, "y": y, "direction_vector": [dx, dy], "pocketed": False}}
        self.balls = {}
        self.pocket_coordinates = pocket_coordinates

    def __call__(self, detection_results: ultralytics.engine.results.Results) -> dict[int, list[int]]:
        """
        Predict the next position of the balls using linear extrapolation.

        :param detection_results: Model detection results
        :return: Dictionary of ball predictions
        """

        self.update_ball_states(detection_results)
        return self.get_ball_predictions()

    def get_ball_predictions(self) -> dict[int, list[int]]:
        """
        Predict the next position of the balls using linear extrapolation.

        :return: Dictionary of ball predictions
        """
        ball_predictions = {}
        for ball_id, ball in self.balls.items():
            if ball["pocketed"]:
                continue
            ball_predictions[int(ball_id)] = self.__predict_ball_position(ball)
        return ball_predictions

    # TODO: This method does not consider the direction of the ball
    # TODO: This method does not consider other balls in the way
    @staticmethod
    def __check_circle_line_intersection(
        x1: float, y1: float, x2: float, y2: float, cx: float, cy: float, cr: float
    ) -> bool:
        """
        Check if a line intersects a circle.

        :param x1: X coordinate of the first point of the line.
        :param y1: Y coordinate of the first point of the line.
        :param x2: X coordinate of the second point of the line.
        :param y2: Y coordinate of the second point of the line.
        :param cx: X coordinate of the center of the circle.
        :param cy: Y coordinate of the center of the circle.
        :param cr: Radius of the circle.
        :return: True if the line intersects the circle, False otherwise.
        """
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            return abs(x1 - cx) <= cr
        m = dy / dx
        c = y1 - m * x1

        # Calculate the distance from the center of the circle to the line
        distance = abs(m * cx - cy + c) / np.sqrt(m**2 + 1)

        # If the distance is less or equal than the radius of the circle, the line intersects or touches the circle
        return distance <= cr

    def get_potential_pots(self, ball_ids: list[int]) -> list[int]:
        potential_pots = []
        for ball_id in ball_ids:
            ball = self.balls[int(ball_id)]
            for pocket_coords in self.pocket_coordinates:
                if self.__check_circle_line_intersection(
                    ball["x"],
                    ball["y"],
                    ball["x"] + ball["direction_vector"][0],
                    ball["y"] + ball["direction_vector"][1],
                    pocket_coords[0],
                    pocket_coords[1],
                    10,
                ):
                    potential_pots.append(int(ball_id))
        return potential_pots

    def update_ball_states(self, detections: ultralytics.engine.results.Results) -> None:
        detections = detections.boxes

        if detections.id is None:
            return

        for i in range(len(detections.id)):
            ball_id = int(detections.id[i])
            ball_x = float(detections.xywh[i][0])
            ball_y = float(detections.xywh[i][1])
            if ball_id not in self.balls:
                self.balls[ball_id] = {
                    "x": ball_x,
                    "y": ball_y,
                    "direction_vector": [0, 0],
                    "pocketed": False,
                }

            self.balls[ball_id]["direction_vector"] = self.__calculate_direction_vector(ball_id, ball_x, ball_y)
            self.balls[ball_id]["x"] = ball_x
            self.balls[ball_id]["y"] = ball_y

    def __calculate_direction_vector(self, ball_id: int, next_x: float, next_y: float) -> list[float]:
        ball = self.balls[ball_id]

        r1 = np.array([ball["x"], ball["y"]])  # Initial position (x1, y1)
        r2 = np.array([next_x, next_y])  # Final position (x2, y2)

        displacement = r2 - r1
        if not displacement.any():
            return displacement
        # Apply a threshold to the displacement to avoid noise in the direction vector
        if np.linalg.norm(displacement) < 0.5:
            return [0.0, 0.0]
        magnitude = np.linalg.norm(displacement)

        # Normalize displacement vector to obtain direction vector
        direction_vector = displacement / magnitude

        return direction_vector

    @staticmethod
    def __predict_ball_position(ball: dict) -> list[int]:
        """
        Predict the next position of a ball using linear extrapolation.

        :param ball: Ball state.
        :return: Predicted the position of the ball.
        """
        return [
            ball["x"] + 100 * ball["direction_vector"][0],
            ball["y"] + 100 * ball["direction_vector"][1],
        ]

    def draw_ball_direction_lines(self, detection_results: ultralytics.engine.results.Results, frame) -> None:
        path_predictions = self.get_ball_predictions()
        for i, ball_id in enumerate(detection_results.boxes.id):
            ball_id = int(ball_id)
            if ball_id not in path_predictions:
                continue
            predicted_ball_coord = path_predictions[ball_id]
            cv2.line(
                frame,
                (int(detection_results.boxes.xywh[i][0]), int(detection_results.boxes.xywh[i][1])),
                (int(predicted_ball_coord[0]), int(predicted_ball_coord[1])),
                (0, 255, 0),
                2,
            )
