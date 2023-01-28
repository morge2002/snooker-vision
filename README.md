# snooker-vision

This is a computer vision project aiming to track pool/snooker balls and the table in real time.
The [YOLOv5 model](https://ultralytics.github.io) is being used as the base model. It is then trained on pool and
snooker images to detect different cue sport related objects.

The final aim of this project is to produce a system that can read a video feed of a real game of pool/snooker and
report back certain details such as score, player at the table, pot success etc. This system can also be used to create
new interactive games on pool and snooker tables. Such as, displaying each player and their number of lives when playing
a game of 'Killer'.

# File Index

- [Get Images Script](file://snooker-vision/object_detection_model/images/get_images.py)
- [Open Images Dataset](datasets/snooker_vision)
- [RoboFlow Dataset](snooker-vision/datasets/robflow_dataset)
- [RoboFlow Dataset (model)](file://object_detection_model/YOLOv5/yolov5/data/snooker_vision_dataset)


- [Training Dataset YAML](file://object_detection_model/YOLOv5/yolov5/data/snooker_vision.yaml)
- [Trained Weights (exp5)](file://object_detection_model/YOLOv5/yolov5/runs/train/exp5/weights/best.pt) 
- [Inference Results (9 ball video)](file://object_detection_model/YOLOv5/yolov5/runs/detect/exp8/9_ball_video.mp4)

# Dataset (RoboFlow)
## Dataset Insights

The dataset used to train the model was downloaded
from [Google's Open Images](https://storage.googleapis.com/openimages/web/download.html). Only the 'Billiard table'
class was used when downloading from this api. Meaning that all images in the dataset have a billiard table. When
exporting the dataset only the 'Billiard table' and 'Ball' classes were used. So, the dataset used to train the model
only contains these classes.

Since the dataset is from Google's Open Images' api we have a nice split between the training and validation set.

### Concerns

This dataset is heavily biased towards American pool tables and balls. The angles at which the photos were taken are
usually from the perspective of someone taking them. This does not suite the intended use case since we are mainly
concerned with object detection from a bird's eye view.

The 'Ball' class is very inconsistent. Both singular and groups of balls are both classified in the
same way. This needs to be changed. Either by labelling each ball in a group or by adding another class (Ball group).

Also, with only 168 images in the training dataset, the dataset is very small. Significantly more images need to be
added, especially of English pool and Snooker.

## Importing the Dataset

1. Create a RoboFlow account and workspace
2. Create a new dataset
3. If you don't have the datasets, run:
    ```python
    pull_store_datasets(
        ["validation", "train", "test"],
        ["Billiard table"],
        ["Ball", "Billiard table"]
    )
    ```
   This will pull the datasets from Open Images and store them on your machine.
4. Upload the images and labels to the RoboFlow dataset
5. Continue with the default setting
6. Generate the dataset
7. Export to YOLOv5 PyTorch (txt)
8. Download the dataset and copy it into the repo.

# The Model
## Training
The following command will train the model on the custom dataset.

- **img** - This is the size of the images. When creating the dataset in RoboFlow it should already format the images to
  640x640.
- **batch** - This is the sample size of images that are passed to the model for each epoch.
- **epochs** - This is the number of epochs. An epoch is an iteration of training. So, when the model is trained on the
  batch of images, that is one epoch. An epoch of 20 means that the model is trained on a batch of images 20 times.
- **data** - This is the path to the dataset yaml. This yaml file specifies the classes in the dataset and the path to
  each split of images.
- **weights** - This is the path to the starting weights of the model. YOLOv5 has provided default weights. Since this a
  transformer model, we can give it base weights that dramatically speed up the training process.

```shell
python train.py --img 640 --batch 32 --epochs 100 --data snooker_vision.yaml --weights yolov5s.pt
```

## Usage (Inference)

### Inference Command

- **<path/to/weights/best.pt>** - The path to the training weights. This will be located in
  yolov5/data/runs/train/exp[n]
  /best.pt
- **<path/to/source>** - The path to the media that you want to run the model on. This can be in
  many [formats](https://ultralytics.github.io/quick-start/#from-your-cloned-repository) such as a video, picture,
  webcam etc.

```shell
python detect.py --weights <path/to/weights/best.pt> --source <path/to/source>
```

The results of the inference will be saved in the yolov5/data/runs/detect directory.

# Results

## 01/28/23

### Pool

[](object_detection_model/YOLOv5/yolov5/runs/detect/exp8/9_ball_video.mp4)

# Next Steps
