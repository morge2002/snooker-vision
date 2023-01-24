"""
Test object detection for snooker/pool table and balls with Google's open images datasets
"""

import fiftyone as fo


# import torch

# train_dataset = fo.zoo.load_zoo_dataset(
#     "open-images-v6",
#     split="train",
#     label_types=["detections"],
#     classes=["Billiard table"],
# )
# print("dataset loaded. Exporting dataset...")
# train_dataset.export(
#     export_dir=os.path.join(os.path.dirname(__file__), "datasets/snooker_vision/train"),
#     dataset_type=fo.types.YOLOv5Dataset,
#     label_field=["Ball", "Billiard table"],
# )


def pull_store_datasets(
    splits: list[str], pull_classes: list[str], export_classes: list[str]
) -> None:
    for split in splits:
        print(f"Pulling {split} dataset...")
        dataset = fo.zoo.load_zoo_dataset(
            "open-images-v6",
            split=split,
            label_types=["detections"],
            classes=pull_classes,
        )
        print(f"Computing {split} dataset metadata...")
        dataset.compute_metadata()
        print(f"Evaluating {split} dataset detections...")
        dataset.evaluate_detections(
            "predictions",
            gt_field="detections",
            method="open-images",
            classes=export_classes,
            use_boxes=True,
        )
        print(f"Exporting {split} dataset...")
        dataset.export(
            export_dir="datasets/snooker_vision",
            dataset_type=fo.types.YOLOv5Dataset,
            split=split,
            classes=export_classes,
        )
        print(f"{split} dataset complete.")


def load_dataset(dataset_dir: str, name: str):
    return fo.Dataset.from_dir(
        dataset_dir=dataset_dir, dataset_type=fo.types.YOLOv5Dataset, name=name
    )
    # return fo.load_dataset(name)


pull_store_datasets(
    ["validation", "train"], ["Billiard table"], ["Ball", "Billiard table"]
)

validation_data = load_dataset("datasets/snooker_vision/", "validation")
validation_data = load_dataset("", name="v")

#
# test_dataset = fo.zoo.load_zoo_dataset(
#     "open-images-v6",
#     split="test",
#     label_types=["detections"],
#     classes=["Billiard table"],
# )
# print("dataset loaded. Exporting dataset...")
# test_dataset.export(
#     export_dir=os.path.join(os.path.dirname(__file__), "datasets/snooker_vision/test"),
#     dataset_type=fo.types.YOLOv5Dataset,
#     label_field=["Ball", "Billiard table"],
# )
#
# validation_dataset = fo.zoo.load_zoo_dataset(
#     "open-images-v6",
#     split="validation",
#     label_types=["detections"],
#     classes=["Billiard table"],
# )
# print("dataset loaded. Exporting dataset...")
# validation_dataset.compute_metadata()
# validation_dataset.evaluate_detections(
#     "predictions",
#     gt_field="detections",
#     method="open-images",
#     classes=["Ball", "Billiard table"],
#     use_boxes=True,
# )
# validation_dataset.export(
#     # export_dir=os.path.join(
#     #     os.path.dirname(__file__), "datasets/snooker_vision/validation"
#     # ),
#     export_dir="datasets/snooker_vision",
#     dataset_type=fo.types.YOLOv5Dataset,
#     # label_field=["Ball", "Billiard table"],
#     split="validation",
#     classes=["Ball", "Billiard table"],
# )

# train_results = train_dataset.evaluate_detections(
#     "predictions",
#     gt_field="detections",
#     method="open-images",
#     classes=["Ball", "Billiard table"],
#     use_boxes=True,
# )
