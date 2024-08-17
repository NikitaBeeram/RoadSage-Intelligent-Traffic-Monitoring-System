import cv2
import os

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

# URL of the video stream from the IP camera app
video_url = "http://192.168.199.210:8080/video"  # Replace with your IP camera URL
cap = cv2.VideoCapture(video_url)

# Check if the video stream was opened correctly
if not cap.isOpened():
    raise Exception("Error opening video stream")

# Get original resolution of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

# Define Region of Interest (ROI) with 50 pixels from the bottom and 100 pixels in height
roi_height = 300
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

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break      
    # Draw the box for the ROI
    cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 255), 2)
    img_roi = img[roi_y: roi_y + roi_height, roi_x: roi_x + roi_width]
    img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,  # Increase to reduce false positives
        minSize=(30, 30),  # Set a minimum size to avoid small, erroneous detections
        maxSize=(300, 300)  # Set a maximum size for detections
    )

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

    # Display the video frame with detected plates
    cv2.imshow("Result", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
