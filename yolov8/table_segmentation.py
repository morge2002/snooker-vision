"""
Crop an image to the table region only using OpenCV
"""
import cv2
import numpy as np
from ultralytics import YOLO

model_path = "runs/detect/train13/weights/best_openvino_model"

# Load trained model
model = YOLO(model_path)

# Open the image file
# image_path = "test_data/snooker_photo.png"
# image_path = "test_data/pool_table_start_position.jpg"
# image_path = "test_data/ultimate_pool_table.png"
# image_path = "test_data/pool_table_start_position.png"

image_path = "test_data/pool_table_top_down.jpg"

# image_path = "test_data/snooker_table_start_layout.png"

results = model(image_path)
# Visualize the results on the frame
annotated_frame = results[0].plot(labels=True, conf=False)
# Display the annotated frame
cv2.imshow("YOLOv8 Inference", annotated_frame)


# Crop the image to the table region
def get_bbox(results, label_index):
    """Get the bounding box for a specific label index"""
    for r in results:
        for i, name in enumerate(r.boxes.cls):
            if name == label_index:
                return r.boxes[i].xyxy[0]
    return None


def get_billiard_table_bbox(inference_results):
    """Get the bounding box for the billiard table"""
    table_label = list(results[0].names.keys())[list(results[0].names.values()).index("Billiard table")]
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


def find_billiard_table_colour(image):
    # Convert the image from BGR to HSV
    image_hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)

    # Reshape the image to a 2D array of pixels
    pixels = image_hsv.reshape((-1, 3))

    # Convert to float32
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5  # Number of clusters
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

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
    hue_threshold = 20
    saturation_threshold = 70
    value_threshold = 70

    lower_bound = colour - np.array(
        [hue_threshold, saturation_threshold, value_threshold]
    )  # Adjust this threshold as needed
    upper_bound = colour + np.array(
        [hue_threshold, saturation_threshold, value_threshold]
    )  # Adjust this threshold as needed

    hsv_ranges = [[0, 0, 0], [179, 255, 255]]

    for i in range(3):
        lower_bound[i] = max(hsv_ranges[0][i], lower_bound[i])
        upper_bound[i] = min(hsv_ranges[1][i], upper_bound[i])

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
    return mask


def find_max_area_contour(img_mask):
    # Find the contours of the mask
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("Number of contours:", len(contours))
    # print("Contour areas:", [cv2.contourArea(c) for c in contours])
    # print(hierarchy)
    # Draw the contours on the original image

    # Find the largest contour
    max_area_contour = max(contours, key=cv2.contourArea)

    return max_area_contour, contours


def table_projection(original_image, table_corners):
    # Define the destination points of the rectangle (4 corners)
    # Aspect ratio of 8-ball, 9-ball, and snooker table is 2:1
    dst_width = 320  # Desired width of the rectangle
    dst_height = 640  # Desired height of the rectangle
    dst_points = np.array(
        [[0, 0], [dst_width, 0], [dst_width, dst_height], [0, dst_height]],
        dtype=np.float32,
    )
    table_corners = np.array(table_corners, dtype=np.float32)
    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(table_corners, dst_points)

    # Apply the perspective transformation
    output_img = cv2.warpPerspective(original_image.copy(), M, (dst_width, dst_height))
    return output_img


def find_billiard_table_corners(table_mask, table_image):
    # Find the largest contour
    max_area_contour, contours = find_max_area_contour(table_mask)
    cv2.drawContours(table_image, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Contours", table_image.copy())

    # Use OpenCV boundingRect function to get the details of the contour
    x, y, w, h = cv2.boundingRect(max_area_contour)
    # cv2.rectangle(table_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rect = cv2.minAreaRect(max_area_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(table_image, [box], 0, (255, 0, 0), 2)

    approx_table_corners = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)

    # Contour approximation
    # TODO: The 'best' epsilon value should be determined automatically but we just get the first one that has 4 corners
    # Iterative Edge Doubling (IED) to find best epsilon
    # epsilon = 0.001
    # max_iterations = 15
    # for _ in range(max_iterations):
    for epsilon in np.linspace(0.001, 0.1, 15):
        perimeter = cv2.arcLength(max_area_contour, True)
        approx = cv2.approxPolyDP(max_area_contour, epsilon * perimeter, True)
        if len(approx) == 4:
            approx_table_corners = approx.reshape(4, 2)
            # Reorder the corners
            table_tolerance = 10
            for i in range(len(approx_table_corners) - 1):
                print(f"iteration {i}: {approx_table_corners}")
                # Assume corners are in order (TL, TR, BR, BL)
                # Lowest (TL), Highest (BR), Highest x and low y (TR), Highest y and low x (BL)
                # Reorder if needed
                for j in range(i + 1, len(approx_table_corners)):
                    if i == 0:  # TL
                        # if the sum coords is greater and the x coord in smaller
                        if (
                            sum(approx_table_corners[i])
                            > sum(approx_table_corners[j])
                            # and approx_table_corners[i][0] > approx_table_corners[j][0]
                        ):
                            tmp = approx_table_corners[i].copy()
                            approx_table_corners[i] = approx_table_corners[j]
                            approx_table_corners[j] = tmp
                    elif i == 1:  # TR
                        # find the corner that has the lowest y coord out of TR, BL, BR
                        if approx_table_corners[i][1] > approx_table_corners[j][1]:
                            tmp = approx_table_corners[i].copy()
                            approx_table_corners[i] = approx_table_corners[j]
                            approx_table_corners[j] = tmp
                    elif i == 2:
                        # find the corner that has the greatest x coord out of BL, BR
                        if approx_table_corners[i][0] < approx_table_corners[j][0]:
                            tmp = approx_table_corners[i].copy()
                            approx_table_corners[i] = approx_table_corners[j]
                            approx_table_corners[j] = tmp
            # approx_table_corners = approx_table_corners.astype(np.float32)
            approx_table_corners = np.float32(
                [
                    approx_table_corners[0] + [-table_tolerance, -table_tolerance],  # Top-left
                    approx_table_corners[1] + [table_tolerance, -table_tolerance],  # Top-right
                    approx_table_corners[2] + [table_tolerance, table_tolerance],  # Bottom-right
                    approx_table_corners[3] + [-table_tolerance, table_tolerance],  # Bottom-left
                ]
            )
            print(f"Approximate corners: {approx_table_corners}")
            print(f"Approximation with epsilon={epsilon}:", approx)
            cv2.drawContours(table_image, [approx], 0, (255, 0, 0), 3)
            break
        # epsilon *= 2

    return approx_table_corners


image = cv2.imread(image_path)

# Crop the table region
# table_image = crop_billiard_table(image, results)
table_image = image.copy()
table_projection_image = table_image.copy()

# Skip crop for now
# table_image = image.copy()
# table_projection_image = image.copy()

# Find the main color of the table
table_colour = find_billiard_table_colour(table_image)

# Create a mask for the table color
table_image_hsv = cv2.cvtColor(table_image.copy(), cv2.COLOR_BGR2HSV)
table_mask = billiard_table_colour_mask(table_image_hsv.copy(), table_colour)

# Find the corners of the billiard table
approx_table_corners = find_billiard_table_corners(table_mask.copy(), table_image)

# Project the table to a rectangle
table_projection_crop_image = table_projection(table_projection_image.copy(), approx_table_corners)
cv2.imshow("Table Projection", table_projection_crop_image)

# Mask the table projection with black
pocket_mask = billiard_table_colour_mask(table_projection_image.copy(), np.array([20, 20, 20], np.uint8))
cv2.imshow("Pocket Mask", pocket_mask)

# --- Edge detection and line detection ---
# def find_intersections(lines):
#     intersections = []
#     for i in range(len(lines)):
#         for j in range(i + 1, len(lines)):
#             line1 = lines[i][0]
#             line2 = lines[j][0]
#             x1, y1, x2, y2 = line1
#             x3, y3, x4, y4 = line2
#
#             # Calculate intersection point
#             denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
#             if denominator != 0:
#                 px = (
#                     (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
#                 ) / denominator
#                 py = (
#                     (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
#                 ) / denominator
#                 intersections.append((int(px), int(py)))
#     return intersections
#
# def largest_rectangle(intersections):
#     # Convert intersections to numpy array
#     points = np.array(intersections)
#
#     # Find convex hull
#     hull = cv2.convexHull(points)
#
#     # Find the bounding rectangle
#     rect = cv2.minAreaRect(hull)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#
#     return box
#
#
# # Convert the image to grayscale
# table_image_grayscale = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
#
# # Perform edge detection using Canny
# edges = cv2.Canny(table_image_grayscale, 50, 150)  # Adjust threshold values as needed
#
# # Perform line detection using Hough Line Transform
# lines = cv2.HoughLinesP(
#     edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=15
# )  # Adjust parameters as needed
#
# # Draw detected lines on the original image
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(table_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
# # Find intersections of the lines
# intersections = find_intersections(lines)
#
# # Draw intersection points on a separate image
# intersection_image = np.zeros_like(image)
# for intersection in intersections:
#     cv2.circle(intersection_image, intersection, 5, (255, 0, 0), -1)
#
# # Find the largest rectangle enclosing the intersections
# box = largest_rectangle(intersections)
# # Draw the rectangle on the original image
# cv2.drawContours(table_image, [box], 0, (0, 0, 255), 2)


# Display the original image with detected edges and lines
# cv2.imshow("Greyscale Image", table_image_grayscale)
cv2.imshow("Original Image with Edges and Lines", table_image)
# cv2.imshow("Intersection Points", intersection_image)


# Plot the histogram of the table image
# table_image_greyscale = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
# plt.hist(table_image_greyscale.ravel(), 256, [0, 256])
# plt.show()

# Show the cropped image
cv2.waitKey(0)
cv2.destroyAllWindows()
