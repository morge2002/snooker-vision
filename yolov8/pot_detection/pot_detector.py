from __future__ import annotations
import ultralytics

from yolov8.ball import Balls
from yolov8.detection_results import DetectionResults
from yolov8.pockets import Pockets
from yolov8.pot_detection import PocketROIHeuristic, LinearExtrapolationHeuristic


# TODO: Think about if the two methods contradict each other, what happens then? Should the ROI pot not count if
#  the ball was not heading towards a pocket?
class PotDetector:
    """
    Pot Detector class to detect if a ball is potted.

    The Pot Detector uses two methods to detect if a ball is potted:
    1. Pocket ROI Heuristic: Detects if a ball is potted based on the region of interest (ROI) of the pockets.
    2. Linear Extrapolation Heuristic: Predicts the path of the balls and detects if a ball is potted based on the path.
    """

    # The window in which all methods must agree on a pot for 100% confidence
    ideal_pot_window = 10
    # Stale pot window. After this window the confidence of the pot is finalised
    stale_pot_window = 30

    current_timestamp = 0

    def __init__(self, balls: Balls, pockets: Pockets):
        self.balls = balls

        # Dictionary of potential pots
        # Structure: {ball_id: {detection_method: frames_since_detection, ...}, ...}
        self.potential_pots: dict[int, dict[str, int]] = {}

        self.successful_pots: dict[int, tuple] = {}

        # Detects if a ball is potted based on pocket ROIs
        self.pot_detector = PocketROIHeuristic(balls)

        # Predicts the path of balls
        self.path_predictor = LinearExtrapolationHeuristic(balls, pockets)

    def __call__(self, detection_results: DetectionResults, timestamp, frame=None) -> None:
        self.detect_potential_pots(detection_results)

        self.path_predictor.draw_ball_direction_lines(detection_results, frame)

        self.current_timestamp = timestamp

        self.detect_pots()

    def detect_potential_pots(self, detection_results: DetectionResults) -> None:
        roi_pots: list[int] = self.pot_detector(detection_results)
        path_pots: list[int] = self.path_predictor(detection_results)

        self.update_potential_pots(roi_pots, path_pots)

    def update_potential_pots(self, roi_pots: list[int], path_pots: list[int]) -> None:
        for ball_id in roi_pots:
            if ball_id not in self.potential_pots:
                self.potential_pots[ball_id] = {"ROI": 0, "path": -1}
            elif self.potential_pots[ball_id]["ROI"] == -1:
                self.potential_pots[ball_id]["ROI"] = 0

        for ball_id in path_pots:
            if ball_id not in self.potential_pots:
                self.potential_pots[ball_id] = {"path": 0, "ROI": -1}
            elif self.potential_pots[ball_id]["path"] == -1:
                self.potential_pots[ball_id]["path"] = 0

        for ball_id, detection_methods in self.potential_pots.items():
            if detection_methods["ROI"] != -1:
                self.potential_pots[ball_id]["ROI"] += 1
            if detection_methods["path"] != -1:
                self.potential_pots[ball_id]["path"] += 1

    def detect_pots(self) -> None:
        """
        Combining detection methods:
        - 100%: ROI and path detect the same pot within the ideal pot window
        - 75%: ROI and path detect the same pot within the stale pot window
        - 50%: Only path detects the pot within the stale pot window
        - 25%: Only ROI detects the pot within the stale pot window
        """

        for ball_id, detection_methods in self.potential_pots.copy().items():
            # If both methods agree on a pot within the ideal pot window or before the stale window, the pot is
            # finalised
            if (
                0 <= detection_methods["ROI"] <= self.ideal_pot_window
                and 0 <= detection_methods["path"] <= self.ideal_pot_window
            ):
                self.ball_potted(ball_id, 1)
                continue
            elif (
                0 <= detection_methods["ROI"] <= self.stale_pot_window
                and 0 <= detection_methods["path"] <= self.stale_pot_window
            ):
                self.ball_potted(ball_id, 0.75)
                continue

            # If the ball has been potted for more than the stale pot window and only one method has detected it,
            # finalise the pot
            if detection_methods["path"] >= self.stale_pot_window:
                self.ball_potted(ball_id, 0.5)
            elif detection_methods["ROI"] >= self.stale_pot_window:
                self.ball_potted(ball_id, 0.25)

    def ball_potted(self, ball_id: int, confidence: float):
        self.balls.ball_potted(ball_id, confidence)
        self.successful_pots[ball_id] = (confidence, self.current_timestamp)
        del self.potential_pots[ball_id]
        print(f"--- Ball {ball_id} potted with confidence {confidence} at {self.current_timestamp} ---")


# Nabe is a nabe
