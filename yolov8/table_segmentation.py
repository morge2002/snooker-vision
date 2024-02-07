"""
Crop an image to the table region only using OpenCV
"""
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
import cv2

model_path = "runs/detect/train13/weights/best_openvino_model"

# Load trained model
model = YOLO(model_path)

# Open the image file
# image_path = "test_data/snooker_photo.png"
image_path = "test_data/pool_table_start_position.jpg"

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


import cv2
import numpy as np


def find_billiard_table_colour(image):
    # Convert the image from BGR to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Reshape the image to a 2D array of pixels
    pixels = image_hsv.reshape((-1, 3))

    # Convert to float32
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5  # Number of clusters
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Find the most common color
    counts = np.bincount(labels.flatten())
    main_color = centers[np.argmax(counts)]

    print("Main Color (HSV):", main_color)

    # Display the main color
    main_color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    main_color_img[:, :] = main_color
    cv2.imshow("Main Color", cv2.cvtColor(main_color_img, cv2.COLOR_HSV2BGR))
    return main_color


def billiard_table_colour_mask(image_hsv, colour):
    # Create a lower and upper bound for the main color
    lower_bound = colour - np.array([20, 100, 255])  # Adjust this threshold as needed
    upper_bound = colour + np.array([20, 100, 255])  # Adjust this threshold as needed

    # Create the color mask
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    # mask = cv2.inRange(image_hsv, np.array([0, 0, 0]), np.array([179, 255, 255]))

    # Apply dilation to smooth the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

    # Display the mask and the original image
    cv2.imshow("Color Mask", mask)


#
# def find_billiard_table_colour(image):
#     # Convert the image from BGR to RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Reshape the image to a 2D array of pixels
#     pixels = image_rgb.reshape((-1, 3))
#
#     # Convert to float32
#     pixels = np.float32(pixels)
#
#     # Define criteria and apply kmeans()
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#     k = 5  # Number of clusters
#     _, labels, centers = cv2.kmeans(
#         pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
#     )
#
#     # Convert back to 8 bit values
#     centers = np.uint8(centers)
#
#     # Find the most common color
#     counts = np.bincount(labels.flatten())
#     main_color = centers[np.argmax(counts)]
#
#     print("Main Color (RGB):", main_color)
#
#     # Display the main color
#     main_color_img = np.zeros((100, 100, 3), dtype=np.uint8)
#     main_color_img[:, :] = main_color
#     cv2.imshow("Main Color", main_color_img)
#     return main_color
#
#
# def billiard_table_colour_mask(image_rgb, colour):
#     # Create a lower and upper bound for the main color
#     lower_bound = colour - np.array([30, 30, 30])  # Adjust this threshold as needed
#     upper_bound = colour + np.array([25, 25, 25])  # Adjust this threshold as needed
#
#     # Create the color mask
#     mask = cv2.inRange(image_rgb, lower_bound, upper_bound)
#
#     # Apply dilation to smooth the mask
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.erode(mask, kernel, iterations=1)
#     mask = cv2.dilate(mask, kernel, iterations=1)
#
#     # Apply the mask to the original image
#     result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
#
#     # Display the mask and the original image
#     cv2.imshow("Color Mask", mask)


image = cv2.imread(image_path)

# Crop the table region
table_image = crop_billiard_table(image, results)

# Find the main color of the table
table_colour = find_billiard_table_colour(table_image)

# Create a mask for the table color
billiard_table_colour_mask(table_image, table_colour)


# --- Edge detection and line detection ---
def find_intersections(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2

            # Calculate intersection point
            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denominator != 0:
                px = (
                    (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
                ) / denominator
                py = (
                    (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
                ) / denominator
                intersections.append((int(px), int(py)))
    return intersections


def largest_rectangle(intersections):
    # Convert intersections to numpy array
    points = np.array(intersections)

    # Find convex hull
    hull = cv2.convexHull(points)

    # Find the bounding rectangle
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    return box


# Convert the image to grayscale
table_image_grayscale = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)

# Perform edge detection using Canny
edges = cv2.Canny(table_image_grayscale, 50, 150)  # Adjust threshold values as needed

# Perform line detection using Hough Line Transform
lines = cv2.HoughLinesP(
    edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=15
)  # Adjust parameters as needed

# Draw detected lines on the original image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(table_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Find intersections of the lines
intersections = find_intersections(lines)

# Draw intersection points on a separate image
intersection_image = np.zeros_like(image)
for intersection in intersections:
    cv2.circle(intersection_image, intersection, 5, (255, 0, 0), -1)

# Find the largest rectangle enclosing the intersections
box = largest_rectangle(intersections)
# Draw the rectangle on the original image
cv2.drawContours(table_image, [box], 0, (0, 0, 255), 2)


# Display the original image with detected edges and lines
cv2.imshow("Greyscale Image", table_image_grayscale)
cv2.imshow("Original Image with Edges and Lines", table_image)
cv2.imshow("Intersection Points", intersection_image)


# Plot the histogram of the table image
# table_image_greyscale = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
# plt.hist(table_image_greyscale.ravel(), 256, [0, 256])
# plt.show()

# Show the cropped image
cv2.imshow("Table", table_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
