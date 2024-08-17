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
from keras.models import load_model  # for loading the model
from main_util import preprocess_input

# Load the saved model
model = load_model("model.h5")

# Process test video frames
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


from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
# Preprocess the images
test_image = preprocess_input(test_image, data_format=None)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 
# Extract features from the images using pretrained model
test_image = base_model.predict(test_image, verbose=0)
test_image.shape

test_image = test_image.reshape(9, 7*7*512)

# Zero center images
test_image = test_image/test_image.max()

# Make predictions
predictions = model.predict(test_image, verbose=0)
'''
print(predictions)

for i in range (0,9):
    if predictions[i][0]<predictions[i][1]:
        print("No Accident")
    else:
        print("Accident")
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
cap = cv2.VideoCapture('he4.mp4')
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