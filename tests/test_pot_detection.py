import unittest

from yolov8.cue_vision import CueVision

english_pool_dish_one_frame_pots = [2060, 5530, 30030, 42460, 57420, 71410, 81350, 94080, 101670]


class TestPotDetection(unittest.TestCase):
    test_video_path = "./test_data/english_pool_dish_one_frame.mp4"
    model_path = "../yolov8/runs/detect/train13/weights/best_openvino_model"
    show_video = False
    pot_milliseconds_threshold = 2000

    def setUp(self):
        self.model = CueVision(model_path=self.model_path)

    def test_detect_pot(self):
        pots: dict[int, tuple] = self.model(self.test_video_path, show_video=self.show_video)
        print(pots)
        true_positives = 0
        false_positives = 0
        # Identify successful detections if a detection timestamp is within 1 second of the ground truth timestamps
        detected_pots = [pot[1] for pot in pots.values()]
        for pot_timestamp in english_pool_dish_one_frame_pots:
            success = False
            for detected_ball_timestamp in detected_pots:
                if abs(detected_ball_timestamp - pot_timestamp) <= self.pot_milliseconds_threshold:
                    true_positives += 1
                    success = True
                    break
            if not success:
                false_positives += 1

        print(f"Pots detected: {len(pots)}")
        print(f"True positives: {true_positives}")
        print(f"False positives: {false_positives}")
