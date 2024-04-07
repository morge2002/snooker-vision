from __future__ import annotations
import ultralytics

from yolov8.ball import Balls
from yolov8.pockets import Pockets
from yolov8.pot_detection import PocketROIHeuristic, LinearExtrapolationHeuristic


class PotDetector:
    def __init__(self, balls: Balls, pockets: Pockets):
        self.balls = balls

        # Detects if a ball is potted based on pocket ROIs
        self.pot_detector = PocketROIHeuristic(balls, pockets)

        # Predicts the path of balls
        self.path_predictor = LinearExtrapolationHeuristic(balls, pockets)

    def __call__(self, detection_results: ultralytics.engine.results.Results, frame=None) -> None:
        # Detect pot using ROI heuristic
        self.pot_detector(detection_results)
        print(f"balls potted: {self.pot_detector.balls_potted}")

        # Predict the next position of the balls using linear extrapolation and show the path on the frame
        self.path_predictor(detection_results)
        self.path_predictor.draw_ball_direction_lines(detection_results, frame)
        print(f"balls towards pockets: {self.path_predictor.get_potential_pots(detection_results.boxes.id)}")


# Nabe is a nabe
