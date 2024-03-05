from __future__ import annotations

import numpy as np
import ultralytics.engine.results


# TODO: Need to add logic to specify if a ball is headed to a pocket. The PotDetector will use this along with other
#  heuristics to determine if a pot has been made
class LinearExtrapolationHeuristic:
    def __init__(self):
        # Data structure to keep track of the balls
        # Structure: {ball_id: {"x": x, "y": y, "direction_vector": [dx, dy], "pocketed": False}}
        self.balls = {}

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

    def update_ball_states(self, detections: ultralytics.engine.results.Results) -> None:
        detections = detections[0].boxes

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
            ball["x"] + 50 * ball["direction_vector"][0],
            ball["y"] + 50 * ball["direction_vector"][1],
        ]
