from __future__ import annotations
import numpy as np


class Ball:
    def __init__(self, ball_id: int, x: float, y: float) -> None:
        self.ball_id: int = ball_id
        self.x: float = x
        self.y: float = y
        self.__direction_vector: list[float] = [0, 0]
        self.__direction_vector_is_current: bool = True
        self.pocketed: bool = False
        self.coordinate_history: list[list[float]] = []
        # Number of frames the ball has been missing since the last sighting
        self.missing_frame_count: int = 0

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
        if not displacement.any():
            return displacement
        # Apply a threshold to the displacement to avoid noise in the direction vector
        if np.linalg.norm(displacement) < 1:
            self.__direction_vector = [0.0, 0.0]
            self.__direction_vector_is_current = True
            return [0.0, 0.0]
        magnitude = np.linalg.norm(displacement)

        # Normalize displacement vector to obtain direction vector
        direction_vector = displacement / magnitude

        self.__direction_vector = direction_vector
        self.__direction_vector_is_current = True
        return direction_vector
