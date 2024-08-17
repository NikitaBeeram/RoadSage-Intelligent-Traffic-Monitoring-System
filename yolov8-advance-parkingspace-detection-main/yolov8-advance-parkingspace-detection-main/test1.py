import cv2
import numpy as np
import cvzone
import pickle

cap = cv2.VideoCapture('easy1.mp4')

drawing = False
points = []
current_name = " "

try:
    with open("pickled_content", "rb") as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except (FileNotFoundError, KeyError):
    polylines = []
    area_names = []

def draw(event, x, y, flags, param):
    global points, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))
        polyline = np.array(points, np.int32)
        polylines.append(polyline)
        current_name = input('Area name: ')
        if current_name:
            area_names.append(current_name)
        else:
            area_names.append("Unnamed Area")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame = cv2.resize(frame, (1020, 500))

    for i, polyline in enumerate(polylines):
        polyline = polyline.reshape((-1, 1, 2))
        cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
        cvzone.putTextRect(frame, f'{area_names[i]}', tuple(polyline[0][0]), 1, 1)

    cv2.imshow('FRAME', frame)
    cv2.setMouseCallback('FRAME', draw)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        with open("pickled_content", "wb") as f:
            data = {'polylines': polylines, 'area_names': area_names}
            pickle.dump(data, f)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
