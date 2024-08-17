import cv2
import os
import pytesseract
from datetime import datetime, timedelta

# Set the path for Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Ensure output directories exist
output_dir = "plates"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

text_output_dir = "plate_text"
if not os.path.exists(text_output_dir):
    os.makedirs(text_output_dir)

# Load Haar cascade for license plates
harcascade = "model/plate_number.xml"
plate_cascade = cv2.CascadeClassifier(harcascade)

# Check if the cascade was loaded correctly
if plate_cascade.empty():
    raise Exception("Error loading Haar cascade")

# Load the video from a file
video_file = "sample.mp4"
cap = cv2.VideoCapture(video_file)

# Check if the video file was loaded correctly
if not cap.isOpened():
    raise Exception("Error loading video file")

# Get original resolution of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

# Define Region of Interest (ROI)
roi_height = 100
roi_y = frame_height - 70 - roi_height  # Bottom 150 pixels
roi_x = 0  # Start from the left edge
roi_width = frame_width  # Use full width

# Parameters
half_frame_width = frame_width // 2
half_frame_height = frame_height // 2


cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", half_frame_width, half_frame_height)

min_area = 1000
count = 0

frame_delay = int(1000 / fps)  # Delay in milliseconds

# Store license plate data and their timestamps
processed_numbers = {}
cooldown_period = timedelta(seconds=1)  # 5-second cooldown

# Open a text file to save the extracted plate numbers
text_file_path = f"{text_output_dir}/plate_numbers.txt"
with open(text_file_path, 'a') as text_file:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break  # End of the video or error reading frame
        
        # Draw the box for the ROI
        cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 255), 2)
        # Define and extract ROI
        img_roi = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

        # Detect license plates
        plates = plate_cascade.detectMultiScale(
            img_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            maxSize=(300, 300)
        )

        for (x, y, w, h) in plates:
            area = w * h
            if area > min_area:
                # Adjust coordinates to original frame
                x_full = roi_x + x
                y_full = roi_y + y

                # Save the detected license plate image
                detected_plate = img[y_full:y_full + h, x_full:x_full + w]

                # Convert to grayscale and apply bilateral filter
                gray = cv2.cvtColor(detected_plate, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 10, 20, 20)

                # Use Tesseract to extract text
                text = pytesseract.image_to_string(gray).strip()
                text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '')

                # Get the current timestamp
                current_time = datetime.now()

                # Check if the text is new or outside the cooldown period
                if text not in processed_numbers or (current_time - processed_numbers[text]) > cooldown_period:
                    # Draw rectangle and label on the detected plate
                    cv2.rectangle(img, (x_full, y_full), (x_full + w, y_full + h), (0, 255, 0), 2)
                    cv2.putText(img, "Number Plate", (x_full, y_full - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                    # Log the new text with a timestamp
                    text_file.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}: {text}\n")
                    text_file.flush()  # Ensure data is written to the file

                    # Update the last processed time for this plate
                    processed_numbers[text] = current_time

        # Display the video frame with detected plates
        cv2.imshow("Result", img)

        # Exit if 'q' is pressed
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
