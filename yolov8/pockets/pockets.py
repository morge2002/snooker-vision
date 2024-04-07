from __future__ import annotations

import ultralytics

from yolov8.ball import Balls

# IDEA: Consider moving the ball_roi method into the Ball class and let the Balls class update and keep track of
# which balls are in a roi (the ball would also know itself). This would involve this class being passed into the
# Balls class. Currently, this doesn't make a difference.


class Pockets:
    def __init__(self, balls: Balls, pocket_coordinates: list[list[int, int]], pocket_roi_radii: list[int]):
        self.balls = balls
        self.pocket_coordinates = pocket_coordinates
        #  Sort RIO radii in descending order
        pocket_roi_radii = sorted(pocket_roi_radii, reverse=True)
        pocket_rois = {
            i: {
                "r": radius,
            }
            for i, radius in enumerate(pocket_roi_radii)
        }
        # TODO: Do I create a Pocket class instead of using a dictionary?
        self.pockets = {
            i: {
                "x": pocket_coord[0],
                "y": pocket_coord[1],
                "rois": pocket_rois.copy(),
            }
            for i, pocket_coord in enumerate(pocket_coordinates)
        }

    def __call__(self, detection_results: ultralytics.engine.results.Results):
        if not isinstance(detection_results, ultralytics.engine.results.Results):
            raise ValueError("Detection results must be of type ultralytics.engine.results.Results.")

        detections = detection_results.boxes
        if detections.id is None:
            return

        ball_ids = [int(ball_id) for ball_id in detections.id]
        self.update_balls_in_rois(ball_ids)

    def __ball_roi(self, ball_id: int) -> tuple[int, list[int]] | tuple[()]:
        """
        Check if a ball is in a region of interest.

        :return: The pocket and the ROIs the ball is in or None if the ball is not in any ROI.
        """
        ball_center = (self.balls[ball_id].x, self.balls[ball_id].y)
        for pocket_key, pocket in self.pockets.items():
            rois_present = []
            for roi_key, roi in pocket["rois"].items():
                if abs(ball_center[0] - pocket["x"]) < roi["r"] and abs(ball_center[1] - pocket["y"]) < roi["r"]:
                    rois_present.append(roi_key)
                    continue

                # If the ball is a previous ROI, return the pocket and the ROIs it's in since it can't be in the
                # other pockets
                if rois_present:
                    return pocket_key, rois_present

                # Stop checking if the ball is not in the first ROI since the ROIs are sorted in descending order
                break
        return ()

    def update_balls_in_rois(self, ball_ids: list[int]) -> None:
        """
        Updates the state of the balls in the ROIs.
        """
        for ball_id in ball_ids:
            ball_roi = self.__ball_roi(ball_id)
            self.balls[ball_id].pocket_roi = ball_roi
