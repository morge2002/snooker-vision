from __future__ import annotations
import numpy as np


class Ball:
    def __init__(self, ball_id: int, x: float, y: float) -> None:
        self.ball_id: int = ball_id
        self.x: float = x
        self.y: float = y
        self.__direction_vector: list[float] = [0, 0]
        self.__direction_vector_is_current: bool = True
        self.__velocity = 0
        self.__velocity_is_current: bool = True
        self.pocketed: bool = False
        # History of the ball's coordinates
        self.coordinate_history: list[list[float]] = []
        # Number of frames the ball has been missing since the last sighting
        self.missing_frame_count: int = 0

    def update_position(self, new_x: float, new_y: float) -> None:
        self.__direction_vector_is_current = False
        self.__velocity_is_current = False
        # Only update the direction vector if the position has changed for three frames
        if len(self.coordinate_history) > 0:
            if self.x == new_x == self.coordinate_history[-1][0] and self.y == new_y == self.coordinate_history[-1][1]:
                self.__direction_vector_is_current = True
                self.__velocity_is_current = True

        self.coordinate_history.append([self.x, self.y])
        self.x = new_x
        self.y = new_y
        # Keep only the last 10 coordinates
        if len(self.coordinate_history) > 10:
            self.coordinate_history.pop(0)
        # Reset the missing frame count
        self.missing_frame_count = 0

    def missed_frame(self) -> None:
        self.missing_frame_count += 1

    def get_velocity(self) -> float:
        """
        Calculate the velocity of the ball.

        :return:
        """
        if self.__velocity_is_current:
            return self.__velocity
        if len(self.coordinate_history) == 0:
            return 0

        r1 = np.array(self.coordinate_history[-1])
        r2 = np.array([self.x, self.y])
        displacement = r2 - r1
        if not displacement.any():
            return 0
        magnitude = np.linalg.norm(displacement)
        if magnitude < 1:
            return 0
        self.__velocity = magnitude
        self.__velocity_is_current = True
        return magnitude

    def __calculate_direction_vector(self) -> list[float]:
        """
        Calculate the direction vector of the ball's movement.

        The direction of the ball is only calculated on retrieval of the direction vector. This is to avoid unnecessary
        calculations if the direction vector is not needed.

        The velocity is also calculated and updated since it is needed to calculate the direction vector.

        This function also updates whether the direction vector and velocity are current or not.

        :return: The direction vector of the ball's movement
        """
        if self.__direction_vector_is_current:
            return self.__direction_vector
        if len(self.coordinate_history) == 0:
            self.__direction_vector = [0.0, 0.0]
            self.__direction_vector_is_current = True
            self.__velocity = 0
            self.__velocity_is_current = True
            return [0.0, 0.0]

        r1 = np.array(self.coordinate_history[-1])  # Previous position (x1, y1)
        r2 = np.array([self.x, self.y])  # Final position (x2, y2)

        displacement = r2 - r1

        # If the displacement is zero, the direction vector is zero
        # Apply a threshold to the displacement to avoid noise in the direction vector
        if not displacement.any() or np.linalg.norm(displacement) < 1:
            self.__direction_vector = [0.0, 0.0]
            self.__direction_vector_is_current = True
            self.__velocity = 0
            self.__velocity_is_current = True
            return [0.0, 0.0]

        # Calculate the magnitude of the displacement vector to obtain the velocity
        magnitude = np.linalg.norm(displacement)
        self.__velocity = magnitude
        self.__velocity_is_current = True

        # Normalize displacement vector to obtain direction vector
        direction_vector = displacement / magnitude
        self.__direction_vector = direction_vector
        self.__direction_vector_is_current = True

        return direction_vector

    def get_direction_vector(self) -> list[float]:
        direction_vector = self.__calculate_direction_vector()
        return direction_vector
