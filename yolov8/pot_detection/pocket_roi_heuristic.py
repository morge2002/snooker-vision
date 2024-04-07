from __future__ import annotations

import ultralytics.engine.results

from yolov8.ball import Balls
from yolov8.pockets import Pockets


class PocketROIHeuristic:
    """
    Pocket ROI Heuristic to detect if a ball is potted.

    The heuristic uses the following steps:
    1. Get the current state of the balls (which region of interests they are in).
    2. Find the balls that are not in the current state but are in the previous state.
    3. Check if the balls were in ROIs closer to the pockets (not the largest).
    4. If the balls were in the closest ROIs, they are potted.
    """

    # Number of frames a ball must be missing before it's considered potted
    missing_frame_threshold = 30

    def __init__(self, balls: Balls, pockets: Pockets):
        self.balls = balls
        self.balls_potted: list[int] = []
        self.pockets = pockets.pockets

    def __call__(self, detection_results: ultralytics.engine.results.Results) -> list[int]:
        """
        Detects if a ball is potted and updates the state of the balls.

        :param detection_results: Model detection results
        :return: List of balls potted
        """
        if not isinstance(detection_results, ultralytics.engine.results.Results):
            raise ValueError("Detection results must be of type ultralytics.engine.results.Results.")

        self.detect()
        return self.balls_potted

    def detect(self):
        """
        Detects if a ball is potted.
        """
        for ball_id, ball in self.balls.items():
            # Ball is potted if it is in a roi, is in the last ROI only, is missing for enough frames and not
            # already potted
            if (
                ball.pocket_roi != ()
                and ball.pocket_roi[1] != [0]
                and ball.missing_frame_count > self.missing_frame_threshold
                and ball_id not in self.balls_potted
            ):
                self.balls_potted.append(ball_id)
                print(f"Ball {ball_id} potted in pocket {ball.pocket_roi[0]}")
