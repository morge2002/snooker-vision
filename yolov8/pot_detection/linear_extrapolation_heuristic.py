from __future__ import annotations

from typing import List

import cv2
import numpy as np
import ultralytics.engine.results

from yolov8.ball import Balls, Ball


class LinearExtrapolationHeuristic:
    def __init__(self, balls: Balls, pocket_coordinates: list[list[int, int]] = None, **kwargs):
        self.balls = balls
        self.pocket_coordinates = pocket_coordinates
        self.pocket_radius = 10
        self.missing_frame_threshold = 30
        self.__pot_velocity_threshold = 1

    def __call__(self, detection_results: ultralytics.engine.results.Results) -> dict[int, list[float]]:
        """
        Predict the next position of the balls using linear extrapolation.

        :param detection_results: Model detection results
        :return: Dictionary of ball predictions
        """
        self.potential_pots(detection_results)
        return self.get_ball_predictions()

    def get_ball_predictions(self) -> dict[int, list[float]]:
        """
        Predict the next position of the balls using linear extrapolation.

        :return: Dictionary of ball predictions
        """
        ball_predictions = {}
        for ball_id, ball in self.balls.items():
            if ball.pocketed:
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
                    ball.x,
                    ball.y,
                    ball.x + ball.get_direction_vector()[0],
                    ball.y + ball.get_direction_vector()[1],
                    pocket_coords[0],
                    pocket_coords[1],
                    self.pocket_radius,
                ):
                    potential_pots.append(int(ball_id))
        return potential_pots

    @staticmethod
    def __predict_ball_position(ball: Ball) -> list[float]:
        """
        Predict the next position of a ball using linear extrapolation.

        :param ball: Ball state.
        :return: Predicted the position of the ball.
        """
        direction_vector = ball.get_direction_vector()
        velocity = ball.get_velocity()
        # Show direction line with length proportional to the velocity
        line_length = 0
        if velocity != 0:
            line_length = 5 * velocity
        return [
            ball.x + (line_length * direction_vector[0]),
            ball.y + (line_length * direction_vector[1]),
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

    def potential_pots(self, detection_results: ultralytics.engine.results.Results) -> list[int]:
        balls_detected = [int(ball_id) for ball_id in detection_results.boxes.id]
        missing_balls = list(set(self.balls) - set(balls_detected))
        potential_pots = self.get_potential_pots(missing_balls)
        print(f"Potential pots towards pockets: {potential_pots}")
        for ball_id in potential_pots:
            # If the ball is missing for more than n frames, is not pocketed and has a velocity above the threshold,
            # consider it potted
            if (
                self.balls[ball_id].missing_frame_count > self.missing_frame_threshold
                and not self.balls[ball_id].pocketed
                and self.balls[ball_id].get_velocity() > self.__pot_velocity_threshold
            ):
                self.balls[ball_id].pocketed = True
                print(f"Ball {ball_id} potted")
        return potential_pots
