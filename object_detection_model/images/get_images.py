"""
Test object detection for snooker/pool table and balls with Google's open images datasets
"""

import fiftyone as fo


def pull_store_datasets(
    splits: list[str], pull_classes: list[str], export_classes: list[str]
) -> None:
    """
    Pulls and stores images from Google's open images api.

    :param splits: Splits of images to pull. ["validation", "train", "test"]
    :param pull_classes: List of class names that must be in the images.
    :param export_classes: List of class names that are in the exported images. Any other labels in the images will be
    ignored.

    :Examples:
    Pull all splits from open images with the class of Billiard Table
    >>> pull_store_datasets(
    >>>     ["validation", "train", "test"], ["Billiard table"], ["Ball", "Billiard table"]
    >>> )
    """
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
        # Not sure if this is needed
        # print(f"Evaluating {split} dataset detections...")
        # dataset.evaluate_detections(
        #     "predictions",
        #     gt_field="detections",
        #     method="open-images",
        #     classes=export_classes,
        #     use_boxes=True,
        # )
        print(f"Exporting {split} dataset...")
        dataset.export(
            export_dir="datasets/snooker_vision",
            dataset_type=fo.types.YOLOv5Dataset,
            split=split,
            classes=export_classes,
        )
        print(f"{split} dataset complete.")


def load_dataset(dataset_dir: str, name: str):
    """
    Loads a dataset from disk. There are two ways to pull the dataset. If the dataset is in fo.list_datasets() then
    specify the name and use fo.load_dataset(). Else, specify the path, name and use fo.Dataset.from_dir().

    :param dataset_dir: Path to the dataset
    :param name: Name of the dataset
    """
    return fo.Dataset.from_dir(
        dataset_dir=dataset_dir, dataset_type=fo.types.YOLOv5Dataset, name=name
    )
    # If the dataset is in fo.list_datasets(), then use this command to load dataset
    # return fo.load_dataset(name)

# validation_data = load_dataset("datasets/snooker_vision/", "validation")
