import cv2    
import time
cpt = 0
maxFrames = 500
num = 546
count = 0
cap=cv2.VideoCapture('s4.mp4')
while cpt < maxFrames:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    cv2.imshow("test window", frame) # show image in window
    cv2.imwrite(r"C:\\Users\\HP\\OneDrive\\Desktop\\carnumberplate-main\\images\\numberplate_%d.jpg" %num, frame)
    time.sleep(0.01)
    num += 1
    cpt += 1
    if cv2.waitKey(5)&0xFF==27:
        break
cap.release()   
cv2.destroyAllWindows()