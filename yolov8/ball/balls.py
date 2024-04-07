from __future__ import annotations

from .ball import Ball
import ultralytics


class Balls:
    stale_ball_threshold = 120

    def __init__(self) -> None:
        self.__balls: dict[int, Ball] = {}
        self.__potted_balls: dict[int, Ball] = {}

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

    def update(self, detections: ultralytics.engine.results.Results) -> None:
        detections = detections.boxes

        if detections.id is None:
            return

        for i in range(len(detections.id)):
            ball_id = int(detections.id[i])
            ball_x = float(detections.xywh[i][0])
            ball_y = float(detections.xywh[i][1])
            ball_w = float(detections.xywh[i][2])
            ball_h = float(detections.xywh[i][3])
            if ball_id not in self.__balls:
                self.__balls[ball_id] = Ball(ball_id, ball_x, ball_y, ball_w, ball_h)
                continue
            self.__balls[ball_id].update_position(ball_x, ball_y, ball_w, ball_h)

        for missed_ball_id in set(self.__balls.keys()) - set(detections.id):
            self.__balls[missed_ball_id].missed_frame()
            # Remove balls that are missing for too long
            if self.__balls[missed_ball_id].missing_frame_count > self.stale_ball_threshold:
                self.__balls.pop(missed_ball_id)

    def ball_potted(self, ball_id: int, confidence: float) -> None:
        """
        Mark a ball as potted and move it to the potted balls.

        :param ball_id: ID of potted ball
        :param confidence: Pot confidence
        """
        self.__balls[ball_id].pocketed = True
        self.__balls[ball_id].potted_confidence = confidence
        self.__potted_balls[ball_id] = self.__balls.pop(ball_id)
