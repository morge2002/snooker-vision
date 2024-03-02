from __future__ import annotations

import ultralytics.engine.results


class PocketROIHeuristic:
    def __init__(self, pocket_coordinates: list[list[int]], roi_radii: list[int], **kwargs):
        self.pocket_coordinates = pocket_coordinates

        #  Sort RIO radii in descending order
        roi_radii = sorted(roi_radii, reverse=True)
        pocket_rois = {
            i: {
                "r": radius,
                "balls": [],
            }
            for i, radius in enumerate(roi_radii)
        }
        self.pockets = {
            i: {
                "x": pocket[0],
                "y": pocket[1],
                "rois": pocket_rois.copy(),
                "balls": [],
            }
            for i, pocket in enumerate(pocket_coordinates)
        }

        # Data structure to keep track of which roi each ball is in. Ball id followed by the list of ROIs it's in
        # Structure: {ball_id: (pocket_id, [roi_id, ...])}
        self.ball_states: dict[int, tuple[int, list[int]]] = {}

        self.kwargs = kwargs

    def __call__(self, detection_results: ultralytics.engine.results.Results) -> list[int]:
        if not isinstance(detection_results, ultralytics.engine.results.Results):
            raise ValueError("Detection results must be of type ultralytics.engine.results.Results.")
        current_ball_states = self.current_ball_state(detection_results)
        balls_potted = self.detect(current_ball_states)
        self.ball_states = current_ball_states
        return balls_potted

    @staticmethod
    def __get_ball_center(ball_xywh: list[int]) -> (int, int):
        """
        Get the center of a ball.
        """
        return ball_xywh[0] + (ball_xywh[2] / 2), ball_xywh[1] + (ball_xywh[3] / 2)

    def __ball_in_roi(self, ball_xywh: list[int]) -> tuple[int, list[int]] | None:
        """
        Check if a ball is in a region of interest.

        Returns the pocket and the ROIs the ball is in.
        """
        ball_center = self.__get_ball_center(ball_xywh)
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
        return None

    # TODO: Figure out why the 147 ball is not being detected as in an roi when almost potted
    def current_ball_state(
        self, detection_results: ultralytics.engine.results.Results
    ) -> dict[int, tuple[int, list[int]]]:
        """
        Return the current state of the balls.
        """
        detections = detection_results.boxes
        if not detections.is_track:
            raise ValueError("Balls must be tracked.")

        ball_states: dict[int, tuple[int, list[int]]] = {}
        for i in range(len(detections.id)):
            ball_rois = self.__ball_in_roi(detections.xywh[i])
            if ball_rois:
                ball_states[int(detections.id[i])] = ball_rois
        return ball_states

    def detect(self, current_ball_states: dict[int, tuple[int, list[int]]]) -> list[int]:
        """
        Detects if a ball is potted.
        """
        balls_potted: list[int] = []
        # Get the balls that are not in the current state but are in the previous state
        potential_potted_balls = set(self.ball_states.keys()) - set(current_ball_states.keys())
        for ball_id in potential_potted_balls:
            pocket_id, rois = self.ball_states[ball_id]
            # The ball can only leave the pocket if it's in the last ROI only
            if rois != [0]:
                balls_potted.append(ball_id)
                print(f"Ball {ball_id} potted in pocket {pocket_id}")

        return balls_potted
