"""
Crop an image to the table region only using OpenCV
"""
from ultralytics import YOLO
import cv2

model_path = "runs/detect/train13/weights/best_openvino_model"

# Load trained model
model = YOLO(model_path)

# Open the image file
image_path = "test_data/snooker_photo.png"

results = model(image_path)


def get_bbox(results, label_index):
    """Get the bounding box for a specific label index"""
    for r in results:
        for i, name in enumerate(r.boxes.cls):
            if name == label_index:
                return r.boxes[i].xyxy[0]
    return None


def get_billiard_table_bbox(inference_results):
    """Get the bounding box for the billiard table"""
    table_label = list(results[0].names.keys())[
        list(results[0].names.values()).index("Billiard table")
    ]
    return get_bbox(inference_results, table_label)


def crop_image_to_bbox(image, bbox):
    """Crop an image to a specific bounding box"""
    return image[bbox[1] : bbox[3], bbox[0] : bbox[2]]


def crop_billiard_table(image, results):
    """Crop the billiard table region from an image"""
    # Get the table region bounding box
    table_bbox = get_billiard_table_bbox(results)

    # Crop the table region
    table_bbox = [int(x) for x in table_bbox]
    return crop_image_to_bbox(image, table_bbox)


# Crop the table region
image = cv2.imread(image_path)
table_image = crop_billiard_table(image, results)

# Show the cropped image
cv2.imshow("Table", table_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
