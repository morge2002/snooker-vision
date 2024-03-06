from __future__ import annotations

from .ball import Ball
import ultralytics


class Balls:
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

    def update(self, detections: ultralytics.engine.results.Results) -> None:
        detections = detections.boxes

        if detections.id is None:
            return

        for i in range(len(detections.id)):
            ball_id = int(detections.id[i])
            ball_x = float(detections.xywh[i][0])
            ball_y = float(detections.xywh[i][1])
            if ball_id not in self.__balls:
                self.__balls[ball_id] = Ball(ball_id, ball_x, ball_y)
            self.__balls[ball_id].update_position(ball_x, ball_y)

        for missed_ball in set(self.__balls.keys()) - set(detections.id):
            self.__balls[missed_ball].missed_frame()
            # if self.__balls[missed_ball].missing_frame_count > 10:
            #     self.__balls.pop(missed_ball)
