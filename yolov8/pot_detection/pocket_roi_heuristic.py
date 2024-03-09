from __future__ import annotations

import ultralytics.engine.results


class PocketROIHeuristic:
    """
    Pocket ROI Heuristic to detect if a ball is potted.

    The heuristic uses the following steps:
    1. Get the current state of the balls (which region of interests they are in).
    2. Find the balls that are not in the current state but are in the previous state.
    3. Check if the balls were in ROIs closer to the pockets (not the largest).
    4. If the balls were in the closest ROIs, they are potted.
    """

    balls_potted = []

    def __init__(self, pocket_coordinates: list[list[int, int]], pocket_roi_radii: list[int], **kwargs):
        self.pocket_coordinates = pocket_coordinates

        #  Sort RIO radii in descending order
        pocket_roi_radii = sorted(pocket_roi_radii, reverse=True)
        pocket_rois = {
            i: {
                "r": radius,
            }
            for i, radius in enumerate(pocket_roi_radii)
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
        """
        Detects if a ball is potted and updates the state of the balls.

        :param detection_results: Model detection results
        :return: List of balls potted
        """
        if not isinstance(detection_results, ultralytics.engine.results.Results):
            raise ValueError("Detection results must be of type ultralytics.engine.results.Results.")
        current_ball_states = self.current_ball_state(detection_results)
        balls_potted = self.detect(current_ball_states)
        if balls_potted:
            self.balls_potted += balls_potted
        self.ball_states = current_ball_states
        return balls_potted

    def __ball_in_roi(self, ball_xywh: list[int]) -> tuple[int, list[int]] | None:
        """
        Check if a ball is in a region of interest.

        :return: The pocket and the ROIs the ball is in or None if the ball is not in any ROI.
        """
        ball_center = (int(ball_xywh[0]), int(ball_xywh[1]))
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

    def current_ball_state(
        self, detection_results: ultralytics.engine.results.Results
    ) -> dict[int, tuple[int, list[int]]]:
        """
        Return the current state of the balls.
        """
        detections = detection_results.boxes

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
            # TODO: Need to consider that the ball disappears for more than n frames
            # The ball can only leave the pocket if it's in the last ROI only
            if rois != [0]:
                balls_potted.append(ball_id)
                print(f"Ball {ball_id} potted in pocket {pocket_id}")

        return balls_potted
