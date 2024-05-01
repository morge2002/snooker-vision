from __future__ import annotations

import ultralytics.engine.results

from yolov8.ball import Balls
from yolov8.detection_results import DetectionResults


class PocketROIHeuristic:
    """
    Pocket ROI Heuristic to detect if a ball is potted.

    The heuristic uses the following steps:
    1. Get the current state of the balls (which region of interests they are in).
    2. Check if the balls were in ROIs closer to the pockets (not the largest).
    3. If the balls were in the closest ROIs and are missing, they are potted.
    """

    # Number of frames a ball must be missing before it's considered potted
    missing_frame_threshold = 30

    def __init__(self, balls: Balls):
        self.balls = balls

    def __call__(self, detection_results: DetectionResults) -> list[int]:
        """
        Detects if a ball is potted and updates the state of the balls.

        :param detection_results: Model detection results
        :return: List of balls potted
        """
        balls_potted = self.detect()
        return balls_potted

    def detect(self) -> list[int]:
        """
        Detects if a ball is potted.
        """
        balls_potted: list[int] = []
        for ball_id, ball in self.balls.items():
            # Ball is potted if it is in a roi, is in the last ROI only, is missing for enough frames and not
            # already potted
            if (
                ball.pocket_roi != ()
                and ball.pocket_roi[1] != [0]
                and ball.missing_frame_count > self.missing_frame_threshold
                and not ball.pocketed
            ):
                balls_potted.append(ball_id)
        return balls_potted
