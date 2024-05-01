import os.path
import unittest
import pandas as pd
from yolov8.cue_vision import CueVision

tests = [
    # (
    #     "English Pool Dish",
    #     os.path.join(os.path.dirname(__file__), "./test_data/english_pool_dish_one_frame.mp4"),
    #     [2060, 5530, 30030, 42460, 57420, 71410, 81350, 94080, 101670],
    #     None,
    # ),
    # (
    #     "English Pool Dish 2",
    #     os.path.join(os.path.dirname(__file__), "./test_data/english_pool_clearance_1.mp4"),
    #     [1160, 3210, 20410, 46650, 69950, 84200, 100910, 112680, 128630, 140760],
    #     # ["fast", "medium", "slow", "slow", "fast", "slow", "fast", "fast", "medium", "slow"],
    #     None,
    # ),
    # (
    #     "9 Ball Pool Dish",
    #     os.path.join(os.path.dirname(__file__), "./test_data/9_ball_clearance.mp4"),
    #     [13520, 22840, 30770, 40610, 52270, 66950, 76150, 85650, 95830],
    #     None,
    # ),
    # (
    #     "Snooker 10 Pots",
    #     os.path.join(os.path.dirname(__file__), "./test_data/barton_snooker_clearance_10_pots.mp4"),
    #     [2290, 15260, 31020, 44230, 63050, 89130, 1203150, 141240, 158080, 175050],
    #     # [5, 25, 45, 50],
    #     None,
    # ),
    (
        "Snooker Clearance",
        os.path.join(os.path.dirname(__file__), "./test_data/snooker_clearance_portrait.mov"),
        [
            2000,
            1369,
            21490,
            32910,
            39820,
            48560,
            59570,
            67350,
            73920,
            81590,
            89930,
            94560,
            104990,
            110440,
            119800,
            126020,
            141610,
            145980,
            155790,
            164300,
            171020,
            177280,
            183200,
            189090,
            196640,
            204840,
            211870,
            217030,
            224330,
            230220,
            241120,
        ],
        # [5, 25, 45, 50],
        [5, 25, 50],
    ),
]


class TestPotDetection(unittest.TestCase):
    model_path = os.path.join(os.path.dirname(__file__), "../yolov8/runs/detect/train15/weights/best_openvino_model")
    show_video = True
    pot_milliseconds_threshold = 2000
    results_file = os.path.join(os.path.dirname(__file__), "./results/pot_detection_results.csv")

    # results_df = pd.DataFrame(columns=["Test Video", "True Positives", "False Positives", "Pots Detected"])

    def setUp(self):
        print("Setting up test")
        try:
            self.results_df = pd.read_csv(self.results_file)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.results_df = pd.DataFrame(
                columns=[
                    "Test Video",
                    "Pots Detected",
                    "Actual Pots",
                    "True Positives",
                    "False Positives",
                    "False Negatives",
                    "True Positives Timestamps",
                    "False Positives Timestamps",
                    "False Negatives Timestamps",
                ]
            )

    def test_detect_pot(self):
        print("Running test")
        for test in tests:
            test_name, test_video_path, expected_results, pocket_rois = test
            self.model = CueVision(model_path=self.model_path, pocket_rois=pocket_rois)
            print(f"Testing video: {test_video_path}")
            pots: dict[int, tuple] = self.model(test_video_path, show_video=self.show_video)
            print(pots)
            true_positives: list[float] = []
            false_negatives: list[float] = []
            # Identify successful detections if a detection timestamp is within 1 second of the ground truth timestamps
            detected_pots = [pot[1] for pot in pots.values()]
            for pot_timestamp in expected_results:
                success = False
                for detected_ball_timestamp in detected_pots:
                    if 0 <= (detected_ball_timestamp - pot_timestamp) <= self.pot_milliseconds_threshold:
                        true_positives.append(pot_timestamp)
                        detected_pots.remove(detected_ball_timestamp)
                        success = True
                        break
                if not success:
                    false_negatives.append(pot_timestamp)

            false_positives: list[float] = detected_pots

            print(f"Pots detected: {len(pots)}")
            print(f"True positives: {true_positives}")
            print(f"False positives: {false_positives}")
            print(f"False negatives: {false_negatives}")

            self.results_df = self.results_df.append(
                {
                    "Test Video": test_name,
                    "Pots Detected": len(pots),
                    "Actual Pots": len(expected_results),
                    "True Positives": len(true_positives),
                    "False Positives": len(false_positives),
                    "False Negatives": len(false_negatives),
                    "True Positives Timestamps": true_positives,
                    "False Positives Timestamps": false_positives,
                    "False Negatives Timestamps": false_negatives,
                },
                ignore_index=True,
            )
            self.results_df.to_csv(self.results_file, index=False)
