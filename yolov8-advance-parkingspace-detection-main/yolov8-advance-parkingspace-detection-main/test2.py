import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone

# Load polylines and area names from the pickle file
try:
    with open("pickled_content", "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except (FileNotFoundError, KeyError):
    polylines = []
    area_names = []

# Load class list from coco.txt
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Load YOLO model
model = YOLO('yolov8s.pt')

# Open video capture
cap = cv2.VideoCapture('easy1.mp4')

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
   
    count += 1
    if count % 3 != 0:
       continue

    frame = cv2.resize(frame, (1020, 500))
    frame_copy = frame.copy()
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
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    if len(polylines) != len(area_names):
        print(f"Mismatch between polylines and area_names: {len(polylines)} polylines, {len(area_names)} area_names")
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
                cv2.circle(frame, (cx1, cy1), 5, (255,0,0), -1)
                cv2.polylines(frame,  [polyline], True, (0, 0, 255), 2)
                counter1.append(cx1)
    
    car_count = len(counter1)
    free_space = len(list2) - car_count
    cvzone.putTextRect(frame, f'CARCOUNTER:-{car_count}', (50,60), 2, 2)
    cvzone.putTextRect(frame, f'FREESPACE:-{free_space}', (50,160), 2, 2)

    cv2.imshow('FRAME', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
