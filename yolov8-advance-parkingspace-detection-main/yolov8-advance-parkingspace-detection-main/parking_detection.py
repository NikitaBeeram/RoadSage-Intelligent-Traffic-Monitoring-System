import streamlit as st
import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone
import tempfile

# Function to process the video and yield frames
def process_video(video_path, polylines, area_names, class_list):
    model = YOLO('yolov8counting_trackingvehicles_main\yolov8s.pt')
    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        list1 = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])

            c = class_list[d]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if 'car' in c:
                list1.append([cx, cy])

        if len(polylines) != len(area_names):
            min_length = min(len(polylines), len(area_names))
            polylines = polylines[:min_length]
            area_names = area_names[:min_length]

        counter1 = []
        list2 = []
        for i, polyline in enumerate(polylines):
            list2.append(i)
            polyline = np.array(polyline, dtype=np.int32)
            polyline = polyline.reshape((-1, 1, 2))
            cv2.polylines(frame, [polyline], True, (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0][0]), 1, 1)
            for i1 in list1:
                cx1 = i1[0]
                cy1 = i1[1]
                result = cv2.pointPolygonTest(polyline, ((cx1, cy1)), False)
                if result >= 0:
                    cv2.circle(frame, (cx1, cy1), 5, (255, 0, 0), -1)
                    cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
                    counter1.append(cx1)

        car_count = len(counter1)
        free_space = len(list2) - car_count
        cvzone.putTextRect(frame, f'CARCOUNTER:-{car_count}', (50, 60), 2, 2)
        cvzone.putTextRect(frame, f'FREESPACE:-{free_space}', (50, 160), 2, 2)

        yield frame

    cap.release()

# Streamlit app
st.title("Parking Space Detection System")
st.write("Upload a video file for processing.")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Load pickled data
    try:
        with open("yolov8-advance-parkingspace-detection-main\yolov8-advance-parkingspace-detection-main\pickled_content", "rb") as f:
            data = pickle.load(f)
            polylines, area_names = data['polylines'], data['area_names']
    except (FileNotFoundError, KeyError):
        polylines = []
        area_names = []

    # Load class list from coco.txt
    with open("yolov8-advance-parkingspace-detection-main\yolov8-advance-parkingspace-detection-main\coco.txt", "r") as my_file:
        class_list = my_file.read().split("\n")

    # Create a placeholder for the video frames
    frame_placeholder = st.empty()

    # Process video and display frames in real-time
    for frame in process_video(video_path, polylines, area_names, class_list):
        frame_placeholder.image(frame, channels="BGR")

    st.write("Processing complete!")
