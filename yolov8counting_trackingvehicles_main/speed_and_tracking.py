import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import time
import tempfile

model = YOLO('yolov8counting_trackingvehicles_main\yolov8s.pt')

def main():
    st.title("Vehicle Speed Detection")
    
    # Sidebar options
    #st.sidebar.title("Settings")
    cy1 = st.slider("Line 1 Y-coordinate", 0, 500, 322)
    cy2 = st.slider("Line 2 Y-coordinate", 0, 500, 368)
    offset = 6
    distance = 25
    
    # File upload
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        my_file = open("yolov8counting_trackingvehicles_main\coco.txt", "r")
        data = my_file.read()
        class_list = data.split("\n")

        count = 0
        tracker = Tracker()
        vh_down = {}
        counter = []
        vh_up = {}
        counter1 = []

        stframe = st.empty()
        
        while cap.isOpened():
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
            list = []
                     
            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                c = class_list[d]
                if 'car' in c or 'truck' in c:
                    list.append([x1, y1, x2, y2])
            bbox_id = tracker.update(list)
            for bbox in bbox_id:
                x3, y3, x4, y4, id = bbox
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2

                if cy1 < (cy + offset) and cy1 > (cy - offset):
                    vh_down[id] = time.time()
                if id in vh_down:
                    if cy2 < (cy + offset) and cy2 > (cy - offset):
                        elapsed_time = time.time() - vh_down[id]
                        if counter.count(id) == 0:
                            counter.append(id)
                            a_speed_ms = distance / elapsed_time
                            a_speed_kh = a_speed_ms * 3.6
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                            cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                if cy2 < (cy + offset) and cy2 > (cy - offset):
                    vh_up[id] = time.time()
                if id in vh_up:
                    if cy1 < (cy + offset) and cy1 > (cy - offset):
                        elapsed1_time = time.time() - vh_up[id]
                        if counter1.count(id) == 0:
                            counter1.append(id)
                            a_speed_ms1 = distance / elapsed1_time
                            a_speed_kh1 = a_speed_ms1 * 3.6
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                            cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
            cv2.putText(frame, 'L1', (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
            cv2.putText(frame, 'L2', (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            d = len(counter)
            u = len(counter1)
            cv2.putText(frame, 'going down: ' + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, 'going up: ' + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            # Display the frame with Streamlit
            stframe.image(frame, channels="BGR")

        cap.release()

if __name__ == "__main__":
    main()
