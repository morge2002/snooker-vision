from __future__ import annotations
import ultralytics

from yolov8.ball import Balls
from yolov8.pot_detection import PocketROIHeuristic, LinearExtrapolationHeuristic


class PotDetector:
    def __init__(self, pocket_coordinates: list[list[int, int]], pocket_rois: list[int], balls: Balls, **kwargs):
        self.pocket_coordinates = pocket_coordinates
        self.pocket_rois = pocket_rois
        self.balls = balls

        # Detects if a ball is potted based on pocket ROIs
        self.pot_detector = PocketROIHeuristic(pocket_coordinates, pocket_rois)

        # Predicts the path of balls
        self.path_predictor = LinearExtrapolationHeuristic(balls, pocket_coordinates)

    def __call__(self, detection_results: ultralytics.engine.results.Results, frame=None) -> None:
        # Detect pot using ROI heuristic
        self.pot_detector(detection_results)
        print(f"balls potted: {self.pot_detector.balls_potted}")

        # Predict the next position of the balls using linear extrapolation and show the path on the frame
        self.path_predictor(detection_results)
        self.path_predictor.draw_ball_direction_lines(detection_results, frame)
        print(f"balls towards pockets: {self.path_predictor.get_potential_pots(detection_results.boxes.id)}")
