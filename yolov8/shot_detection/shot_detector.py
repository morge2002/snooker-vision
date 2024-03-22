from yolov8.ball import Balls


class ShotDetector:
    shot_started: bool = False
    # TODO: Will this movement threshold depend on the perspective table
    #  (balls further away will travel less pixels for the same distance?)
    movement_threshold: int = 2
    # Number of frames the shot has lasted/ended for
    shot_change_duration: int = 0
    # Number of frames for a valid shot start/end
    shot_change_threshold: int = 15

    def __init__(self, balls: Balls):
        self.balls = balls

    # TODO: This method should be called for a rolling window of 'n' frames
    #  to ensure the balls have moved or stopped moving
    def __call__(self, detections, *args, **kwargs) -> bool:
        # If the shot has started, then check if it ends and vice versa
        if self.shot_started:
            next_shot_decision = not self.__detect_shot_end(detections)
        else:
            next_shot_decision = self.__detect_shot_start(detections)

        if self.shot_started != next_shot_decision:
            self.shot_change_duration += 1
            if self.shot_change_duration >= self.shot_change_threshold:
                self.shot_started = next_shot_decision
                self.shot_change_duration = 0
        return self.shot_started

    # TODO: Do this for a rolling window of 'n' frames to ensure the cue ball is moving
    def __detect_shot_start(self, detections) -> bool:
        # # Cannot confirm a shot has taken place if the cue ball can't be found
        # cue_ball_id = self.balls.get_cue_ball_id(detections)
        # if cue_ball_id is None:
        #     return False
        #
        # return self.balls[cue_ball_id].get_magnitude() > self.movement_threshold
        detections = detections.boxes
        if not hasattr(detections, "id") or detections.id is None:
            return False

        for ball_id in detections.id:
            # If any ball is moving, then the shot has started
            if self.balls[int(ball_id)].get_magnitude() > self.movement_threshold:
                return True

        return False

    def __detect_shot_end(self, detections) -> bool:
        detections = detections.boxes
        if not hasattr(detections, "id") or detections.id is None:
            return False

        for ball_id in detections.id:
            # If any ball is moving, then the shot has not ended
            if self.balls[int(ball_id)].get_magnitude() > self.movement_threshold:
                return False

        # If no ball is moving, then the shot has ended
        return True
