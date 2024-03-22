from __future__ import annotations

import cv2
import numpy as np

from .ball import Ball
import ultralytics


class Balls:
    cue_ball_id: int | None = None

    def __init__(self) -> None:
        self.__balls: dict[int, Ball] = {}

    def __getitem__(self, ball_id: int) -> Ball:
        return self.__balls[ball_id]

    def __setitem__(self, ball_id: int, ball: Ball) -> None:
        self.__balls[ball_id] = ball

    def __contains__(self, ball_id: int) -> bool:
        return ball_id in self.__balls

    def __iter__(self) -> iter:
        return iter(self.__balls)

    def __len__(self) -> int:
        return len(self.__balls)

    def __str__(self) -> str:
        return str(self.__balls)

    def __repr__(self) -> str:
        return repr(self.__balls)

    def __call__(self) -> dict[int, Ball]:
        return self.__balls

    def items(self):
        return self.__balls.items()

    def update(self, detections: ultralytics.engine.results.Results, frame=None) -> None:
        detections = detections.boxes

        if detections.id is None:
            return

        for i in range(len(detections.id)):
            ball_id = int(detections.id[i])
            ball_x = float(detections.xywh[i][0])
            ball_y = float(detections.xywh[i][1])
            if ball_id not in self.__balls:
                ball_width = float(detections.xywh[i][2])
                ball_height = float(detections.xywh[i][3])
                self.__balls[ball_id] = Ball(ball_id, ball_x, ball_y, frame, ball_width, ball_height)
            self.__balls[ball_id].update_position(ball_x, ball_y)

        for missed_ball in set(self.__balls.keys()) - set(detections.id):
            self.__balls[missed_ball].missed_frame()
            # if self.__balls[missed_ball].missing_frame_count > 10:
            #     self.__balls.pop(missed_ball)

    @staticmethod
    def __colour_is_white(colour: np.array):
        # HSV colour thresholds for white
        lower_threshold = np.array([0, 0, 200], np.uint8)
        higher_threshold = np.array([180, 30, 255], np.uint8)

        result = cv2.inRange(colour, lower_threshold, higher_threshold)

        return all(result)

    def __find_cue_ball(self, ball_ids) -> int | None:
        for ball_id in ball_ids:
            ball_id = int(ball_id)
            ball = self.__balls[ball_id]
            if self.__colour_is_white(ball.colour):
                self.cue_ball_id = ball_id
                return ball_id
        return None

    def get_cue_ball_id(self, detections) -> int | None:
        detections = detections.boxes

        if detections.id is None:
            return None

        if self.cue_ball_id is not None and self.cue_ball_id in detections.id:
            return self.cue_ball_id

        new_ball_ids = set(detections.id) - set(self.__balls.keys())

        return self.__find_cue_ball(new_ball_ids)
