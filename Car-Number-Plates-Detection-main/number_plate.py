import cv2
import os
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Ensure the output directory exists
output_dir = "plates"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load Haar cascade for license plates
model = "Car-Number-Plates-Detection-main\model\plate_number.xml"
plate_cascade = cv2.CascadeClassifier(model)

# Check if the cascade was loaded correctly
if plate_cascade.empty():
    raise Exception("Error loading Haar cascade")

# Load the video from a file
video_file = "Car-Number-Plates-Detection-main/s3.mp4"
cap = cv2.VideoCapture(video_file)

# Check if the video file was loaded correctly
if not cap.isOpened():
    raise Exception("Error loading video file")

# Get original resolution of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

# Define Region of Interest (ROI) with 50 pixels from the bottom and 100 pixels in height
roi_height = 100
roi_y = frame_height - 80 - roi_height  # 50 pixels from the bottom
roi_x = 0  # Start from the left edge
roi_width = frame_width  # Use full width

half_frame_width = frame_width // 2
half_frame_height = frame_height // 2

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", half_frame_width, half_frame_height)

min_area = 1000
count = 0

frame_delay = int(2000 / fps)  # Delay in milliseconds

# Initialize lists to store ground truth and predictions
true_labels = []
predicted_labels = []

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    
    # Draw the box for the ROI
    cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 255), 2)
    img_roi = img[roi_y: roi_y + roi_height, roi_x: roi_x + roi_width]
    img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

    # Detect plates
    plates = plate_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,  # Increase to reduce false positives
        minSize=(30, 30),  # Set a minimum size to avoid small, erroneous detections
        maxSize=(300, 300)  # Set a maximum size for detections
    )

    frame_predicted_labels = []
    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            # Adjust coordinates to original frame
            x_full = roi_x + x
            y_full = roi_y + y

            # Draw rectangle and label on the detected plate
            cv2.rectangle(img, (x_full, y_full), (x_full + w, y_full + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x_full, y_full - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Save the detected license plate image
            detected_plate = img[y_full: y_full + h, x_full: x_full + w]
            filename = f"{output_dir}/scanned_img_{count}.jpg"
            cv2.imwrite(filename, detected_plate)
            count += 1

            frame_predicted_labels.append((x_full, y_full, x_full + w, y_full + h))

    # Add the frame predictions to the overall predictions
    predicted_labels.append(frame_predicted_labels)

    # Display the video frame with detected plates
    cv2.imshow("Result", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Define a function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

# Example ground truth (manually labeled for each frame)
# Format: [(x1, y1, x2, y2), ...]
ground_truth = [...]  # Replace with your actual ground truth data

# Calculate true positives, false positives, and false negatives
true_positives = 0
false_positives = 0
false_negatives = 0

iou_threshold = 0.5  # Adjust as needed

for gt_frame, pred_frame in zip(ground_truth, predicted_labels):
    matched = [False] * len(gt_frame)

    for pred_box in pred_frame:
        match_found = False
        for i, gt_box in enumerate(gt_frame):
            if calculate_iou(pred_box, gt_box) >= iou_threshold:
                match_found = True
                if not matched[i]:
                    true_positives += 1
                    matched[i] = True
                break
        if not match_found:
            false_positives += 1

    for match in matched:
        if not match:
            false_negatives += 1

# Calculate precision, recall, and F1-score
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Accuracy: {accuracy}")

# Measure processing time per frame
processing_times = []
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    start_time = time.time()
    
    img_roi = img[roi_y: roi_y + roi_height, roi_x: roi_x + roi_width]
    img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        maxSize=(300, 300)
    )

    end_time = time.time()
    processing_times.append(end_time - start_time)

cap.release()

average_processing_time = sum(processing_times) / len(processing_times)
print(f"Average Processing Time per Frame: {average_processing_time:.4f} seconds")
