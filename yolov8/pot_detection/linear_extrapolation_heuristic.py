from __future__ import annotations

import cv2
import numpy as np
import ultralytics.engine.results

from yolov8.ball import Balls, Ball


# TODO: Collision detection only takes the line from the centre into account, however, the outer edge could hit
#  the circle. Maybe change this algorithm to see if two circles collide. Although, the size of the object ball
#  will change as it moves (assuming the ball it contacts with is stationary)
class LinearExtrapolationHeuristic:
    def __init__(self, balls: Balls, pocket_coordinates: list[list[int, int]] = None):
        """
        Initialise the class with balls collection and pocket information.

        :param balls: State of all balls (passed by reference)
        :param pocket_coordinates: List of pocket coordinates
        """
        self.balls = balls
        self.pocket_coordinates = pocket_coordinates
        # Size of pockets
        self.pocket_radius = 10
        # The minimum number of frames for a ball to be missing to be considered a pot
        self.missing_frame_threshold = 30
        # The minimum velocity a ball can be travelling before it disappears to be considered a pot
        self.__pot_velocity_threshold = 1

    def __call__(self, detection_results: ultralytics.engine.results.Results) -> dict[int, list[float]]:
        """
        Identify balls potted by identifying balls that are no longer detected, were heading towards a pocket,
        and had reasonable velocity.

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

    @staticmethod
    def __check_circle_line_intersection(
        x1: float, y1: float, x2: float, y2: float, cx: float, cy: float, cr: float
    ) -> bool:
        """
        Check if a line intersects a circle.

        :param x1: Line start point X coordinate.
        :param y1: Line start point Y coordinate.
        :param x2: Line end point X coordinate.
        :param y2: Line end point Y coordinate.
        :param cx: Circle centre X coordinate.
        :param cy: Circle centre Y coordinate.
        :param cr: Radius of the circle.
        :return: True if the line intersects the circle, False otherwise.
        """
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            return abs(x1 - cx) <= cr
        m = dy / dx
        c = y1 - m * x1

        # Calculate the distance from the centre of the circle to the line
        distance = abs(m * cx - cy + c) / np.sqrt(m**2 + 1)

        # If the distance is less or equal than the radius of the circle, the line intersects or touches the circle
        return distance <= cr

    @staticmethod
    def __is_object_in_direction(ball: Ball, object_x: float, object_y: float, object_radius: float) -> bool:
        """
        Check if an object is in the direction that the ball is travelling by seeing if it is in the same quadrant
        based on the direction vector of the ball.

        :Example:
        If ball 1 is moving towards the top right pocket and ball 2 that is in that pocket, then True is returned.
        However, if ball 2 is behind the ball 1, relative to the ball 1's direction, False is returned.

        :param ball: Ball for the object to be checked against
        :param object_x: Object X coordinate
        :param object_y: Object Y coordinate
        :param object_radius: Object radius
        :return: True if the object is within the quadrant of the path of the ball
        """
        direction_vector = ball.get_direction_vector()
        if direction_vector[0] > 0 and object_x + object_radius < ball.x:
            return False
        if direction_vector[0] < 0 and object_x - object_radius > ball.x:
            return False
        if direction_vector[1] > 0 and object_y + object_radius < ball.y:
            return False
        if direction_vector[1] < 0 and object_y - object_radius > ball.y:
            return False
        return True

    def get_potential_pots(self, ball_ids: list[int]) -> list[int]:
        """
        Finds the balls that are heading towards a pocket that is not obstructed by other balls.

        :param ball_ids: List of ball ids to be checked
        :return: List of ball ids that are heading towards a pocket that is not obstructed by other balls.
        """
        potential_pots = []
        # Find balls that are heading towards a pocket
        for ball_id in ball_ids:
            ball = self.balls[int(ball_id)]
            if ball.get_velocity() < self.__pot_velocity_threshold:
                continue
            for pocket_coords in self.pocket_coordinates:
                if not self.__is_object_in_direction(ball, pocket_coords[0], pocket_coords[1], self.pocket_radius):
                    continue

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
                    break

        # Check if the ball is obstructed by other balls
        for ball_id in potential_pots:
            ball = self.balls[ball_id]
            for other_ball_id, other_ball in self.balls.items():
                if other_ball_id == ball_id:
                    continue
                # Average width and height of the ball (halved to get the radius)
                ball_average_radius = (ball.w + ball.h) / 4
                # Check if the other ball is in the direction of the current ball
                if not self.__is_object_in_direction(ball, other_ball.x, other_ball.y, ball_average_radius):
                    continue

                if self.__check_circle_line_intersection(
                    ball.x,
                    ball.y,
                    ball.x + ball.get_direction_vector()[0],
                    ball.y + ball.get_direction_vector()[1],
                    other_ball.x,
                    other_ball.y,
                    ball_average_radius,
                ):
                    potential_pots.remove(ball_id)
                    break
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
        """
        Draw direction lines on the frame.

        :param detection_results: Results from YOLO inference
        :param frame: The frame to draw onto.
        """
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
        """
        Finds balls that could have been potted based on their direction of travel.

        :param detection_results: Results from YOLO inference
        :return: List of ball ids that have been detected as potted
        """
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
