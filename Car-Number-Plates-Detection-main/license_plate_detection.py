import cv2
import numpy as np
import streamlit as st
import tempfile
from PIL import Image

def process_video(video_source, plate_cascade):
    cap = cv2.VideoCapture(video_source)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    roi_height = 100
    roi_y = frame_height - 80 - roi_height
    roi_x = 0
    roi_width = frame_width

    min_area = 1000
    count = 0

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

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

        for (x, y, w, h) in plates:
            area = w * h
            if area > min_area:
                x_full = roi_x + x
                y_full = roi_y + y

                cv2.rectangle(img, (x_full, y_full), (x_full + w, y_full + h), (0, 255, 0), 2)
                cv2.putText(img, "Number Plate", (x_full, y_full - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        yield img

    cap.release()

# Streamlit app
st.title("License Plate Detection")
st.write("Upload a video file or connect to your phone camera for live processing.")

option = st.selectbox("Choose input source", ("Upload Video", "Phone Camera"))

video_source = None

if option == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_source = tfile.name

elif option == "Phone Camera":
    default_ip_address = "http://192.168.199.210:8080/video"  # Default IP address for phone camera
    #ip_address = st.text_input("Enter the IP address of your phone camera (e.g., http://192.168.1.2:8080/video)", default_ip_address)
    ip_address = default_ip_address
    if ip_address:
        video_source = ip_address

if video_source:
    model = "model/plate_number.xml"
    plate_cascade = cv2.CascadeClassifier(model)

    if plate_cascade.empty():
        st.error("Error loading Haar cascade")
    else:
        frame_placeholder = st.empty()

        for frame in process_video(video_source, plate_cascade):
            frame_placeholder.image(frame)

    st.write("Processing complete!")
