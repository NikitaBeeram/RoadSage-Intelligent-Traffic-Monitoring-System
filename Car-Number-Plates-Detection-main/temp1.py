import cv2
import os
import csv
import easyocr
import datetime

# Ensure the output directory exists
output_dir = "plates"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load Haar cascade for license plates
harcascade = "model/plate_number.xml"
plate_cascade = cv2.CascadeClassifier(harcascade)

# Check if the cascade was loaded correctly
if plate_cascade.empty():
    raise Exception("Error loading Haar cascade")

# Load the video from a file
video_file = "s3.mp4"
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

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Open CSV file for writing
csv_file = open('detected_plates.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Plate Text'])  # Write header

# Initialize empty list to store previously detected plates
prev_detected_plates = []

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Draw ROI rectangle
    cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 255), 2)

    img_roi = img[roi_y: roi_y + roi_height, roi_x: roi_x + roi_width]
    img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        maxSize=(300, 300)
    )

    # Initialize list to store detected plates in the current frame
    detected_plates_in_frame = []

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            x_full = roi_x + x
            y_full = roi_y + y

            # Draw rectangle around the detected license plate
            cv2.rectangle(img, (x_full, y_full), (x_full + w, y_full + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x_full, y_full - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Save the detected license plate information to CSV file
            detected_plate = img[y_full: y_full + h, x_full: x_full + w]

            # Perform OCR using EasyOCR
            results = reader.readtext(detected_plate)

            # Extract the text from the OCR results
            plate_text = ""
            for result in results:
                plate_text += result[1] + " "

            # Get timestamp of the frame
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Check if the detected plate is new (not detected in previous frames)
            if plate_text.strip() not in prev_detected_plates:
                # Write plate information to CSV
                csv_writer.writerow([timestamp, plate_text.strip()])
                prev_detected_plates.append(plate_text.strip())  # Add detected plate to list of previously detected plates

                # Save the detected license plate image
                cv2.imwrite(f"{output_dir}/scanned_img_{count}.jpg", detected_plate)
                count += 1

    # Display the video frame with detected plates
    cv2.imshow("Result", img)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

csv_file.close()

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
