"""
Uploads a model to Roboflow for a specific project and dataset version.
"""
import roboflow

model_path = "runs/detect/train11"

roboflow.login()

rf = roboflow.Roboflow()

workspace = rf.workspace("morganlewis")
project = workspace.project("snooker-vision-tjn0z")
version = project.version("1")
version.deploy("yolov8", model_path)
