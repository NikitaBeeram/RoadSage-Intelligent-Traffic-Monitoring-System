
import os
import cv2     # for capturing videos
import math 
import geocoder
import requests
import pandas as pd
from twilio.rest import Client
from geopy.geocoders import Nominatim
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from matplotlib import pyplot as plt 
from skimage.transform import resize   # for resizing images

count = 0
videoFile = "Accidents.mp4"
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
frameRate = cap.get(5) # frame rate

# Create a folder to save the frames
folder_name = "frames"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

while(cap.isOpened()):
    frameId = cap.get(1) # current frame number
    ret, frame = cap.read()
    if not ret:
        break
    if frameId % math.floor(frameRate) == 0:
        filename = os.path.join(folder_name, "%d.jpg" % count)
        cv2.imwrite(filename, frame)
        count += 1

cap.release()
print("Done!")

img = plt.imread('frames\\0.jpg')   # reading image using its name
plt.imshow(img)

data = pd.read_csv('mapping.csv')     # reading the csv file
data.head()


X = [ ]     # creating an empty array
for img_name in data.Image_ID:
    img = plt.imread('frames\\' + img_name)
    X.append(img)  # storing each image in array X
X = np.array(X)    # converting list to array

from tensorflow.keras.utils import to_categorical

# Assuming 'data' is a pandas DataFrame and 'Class' is the column with categorical data
y = data['Class']
dummy_y = to_categorical(y)

image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)

from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X,data_format=None)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)

from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 

X_train = base_model.predict(X_train,verbose=0)
X_valid = base_model.predict(X_valid,verbose=0)
X_train.shape, X_valid.shape

X_train = X_train.reshape(155, 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(67, 7*7*512)
train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()

model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(2, activation='softmax'))    # output layer

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid), verbose=0)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

count = 0
videoFile = "Accident-1.mp4"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="test%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")
test = pd.read_csv('test.csv')

test_image = []
for img_name in test.Image_ID:
    img = plt.imread('' + img_name)
    test_image.append(img)
test_img = np.array(test_image)

test_image = []
for i in range(0,test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)

# preprocessing the images
test_image = preprocess_input(test_image, data_format=None)

# extracting features from the images using pretrained model
test_image = base_model.predict(test_image)
test_image.shape


test_image = test_image.reshape(9, 7*7*512)

# zero centered images
test_image = test_image/test_image.max()

predictions = model.predict(test_image,verbose=0)
'''
print(predictions)

for i in range (0,9):
    if predictions[i][0]<predictions[i][1]:
        print("No Accident")
    else:
        print("Accident")
    

model.save("model.h5")
'''
geoLoc = Nominatim(user_agent="GetLoc")
g = geocoder.ip('me')
locname = geoLoc.reverse(g.latlng)
account_sid = 'ACf63d5cb6bcab785f83dcb33db5356b70'
auth_token = 'b1088d86a401d3a2aa442b7a12d20f94'
client = Client(account_sid, auth_token)

import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from twilio.rest import Client

# Twilio credentials
account_sid = 'ACf63d5cb6bcab785f83dcb33db5356b70'
auth_token = 'b1088d86a401d3a2aa442b7a12d20f94'
twilio_phone_number = '+16365574558'

# Email credentials
smtp_server = 'smtp.gmail.com'
smtp_port = 587
email_user = 'nikijaya1221@gmail.com'
email_password = 'ccjj rkus upym wgri'

# Initialize Twilio client
client = Client(account_sid, auth_token)

def send_sms(body, from_, to):
    message = client.messages.create(
        body=body,
        from_=from_,
        to=to
    )
    return message.sid

def send_email(subject, body, to_email):
    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(email_user, email_password)
        text = msg.as_string()
        server.sendmail(email_user, to_email, text)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
print("Detecting.............")
cap = cv2.VideoCapture('./testvideos/test (1).mp4')
i = 0
flag = 0

while(True):
    ret, frame = cap.read()
    if ret == True:
        if predictions[int(i/15) % 9][0] < predictions[int(i/15) % 9][1]:
            predict = "No Accident"
        else:
            predict = "Accident"
            flag = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, predict, (50, 50), font, 1, (0, 255, 255), 3, cv2.LINE_4)
        cv2.imshow('Frame', frame)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
print("Captured!!")
# Release the cap object
cap.release()
# Close all windows
cv2.destroyAllWindows()

# Send SMS and Email if an accident is detected
if flag == 1:
    locname = "Location"  # Replace with actual location data if available
    message_body = f"Accident detected in {locname}"
    recipient_phone_number = '+918331839758'  # Replace with the actual phone number
    recipient_email = 'nikitabeeram1904@gmail.com'  # Replace with the actual email address

    # Send SMS
    message_sid = send_sms(message_body, twilio_phone_number, recipient_phone_number)
    print(f"Message sent with SID: {message_sid}")
    
    # Send Email
    email_subject = "Accident Detection Alert"
    send_email(email_subject, message_body, recipient_email)
else:
    print("No accident")
