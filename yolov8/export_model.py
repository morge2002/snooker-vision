from ultralytics import YOLO

model = YOLO("./runs/detect/train15/weights/best.pt")
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'

# --- Manual Testing ---
# video_path = "../tests/test_data/barton_snooker_clearance_10_pots.mp4"
# video_path = "../tests/test_data/snooker_clearance.mp4"
# video_path = "../tests/test_data/english_pool_dish_one_frame.mp4"
# video_path = "../tests/test_data/pool_overhead_2.mp4"
# video_path = "../tests/test_data/english_pool_clearance_1.mp4"
# model = YOLO("./runs/detect/train15/weights/best_openvino_model")

# model(video_path, save=True)
# model.track(
#     video_path,
#     persist=True,
#     tracker="track/custom_bytetrack.yaml",
#     device="cpu",
#     max_det=23,
#     save=True,
# )

# Ball only dataset - Train 15
# Batch size 113 for balls model with 150 epoch
# results_dict = {
#     "metrics/precision(B)": 0.9877414815812201,
#     "metrics/recall(B)": 0.9513585282750239,
#     "metrics/mAP50(B)": 0.9822605916065479,
#     "metrics/mAP50-95(B)": 0.8292994762801378,
#     "fitness": 0.8445955878127789,
# }
# speed = {
#     "preprocess": 0.48762162526448566,
#     "inference": 1.5714502334594727,
#     "loss": 6.993611653645833e-05,
#     "postprocess": 1.576865514119466,
# }
