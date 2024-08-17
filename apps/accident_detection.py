import streamlit as st
import cv2
import math
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from skimage.transform import resize
from twilio.rest import Client
from geopy.geocoders import Nominatim
import tempfile
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
from keras.applications.vgg16 import preprocess_input
import os
from keras.applications.vgg16 import VGG16
import asyncio

# Load the saved model
model = load_model('C:\\Users\HP\\OneDrive\\Desktop\\New folder\\accident_detection\\model.h5')

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

def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    frameRate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare a temporary folder to store frames
    temp_dir = tempfile.TemporaryDirectory()
    
    frames = []
    while cap.isOpened():
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % math.floor(frameRate) == 0:
            filename = os.path.join(temp_dir.name, f"frame{int(frameId)}.jpg")
            cv2.imwrite(filename, frame)
            frames.append(filename)
    cap.release()

    return frames, frameRate, temp_dir

def preprocess_frames(frames):
    test_image = []
    for img_path in frames:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        test_image.append(img_array)
    test_image = np.vstack(test_image)
    
    test_image = preprocess_input(test_image)
    
    return test_image

def detect_accidents(model, test_image):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = base_model.predict(test_image, verbose=0)
    features = features.reshape(features.shape[0], -1)
    features = features / features.max()
    predictions = model.predict(features, verbose=0)
    return predictions

def process_video_and_detect_accidents(video_source):
    frames, frameRate, temp_dir = process_video(video_source)
    test_image = preprocess_frames(frames)
    predictions = detect_accidents(model, test_image)
    return frames, frameRate, predictions, temp_dir

def display_results(video_source, frames, frameRate, predictions):
    cap = cv2.VideoCapture(video_source)
    i = 0
    flag = 0
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % math.floor(frameRate) == 0:
            if predictions[int(i/frameRate) % len(predictions)][0] < predictions[int(i/frameRate) % len(predictions)][1]:
                predict = "No Accident"
            else:
                predict = "Accident"
                flag = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, predict, (50, 50), font, 1, (0, 255, 255), 3, cv2.LINE_4)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame)
        i += 1
        time.sleep(1/frameRate)  # Display the frame at the original video frame rate

    cap.release()

    return flag

def run(uploader_key):
    st.title("Accident Detection")
    st.write("Upload a video file for accident detection.")

    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"], key=uploader_key)

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_source = tfile.name

        st.text("Processing video...")
        frames, frameRate, predictions, temp_dir = st.cache_data(process_video_and_detect_accidents)(video_source)
        
        st.text("Displaying results...")
        flag = display_results(video_source, frames, frameRate, predictions)
        
        temp_dir.cleanup()

        # Send SMS and Email if an accident is detected
        if flag == 1:
            locname = "Location"  # Replace with actual location data if available
            message_body = f"Accident detected in {locname}"
            recipient_phone_number = '+918331839758'  # Replace with the actual phone number
            recipient_email = 'nikitabeeram1904@gmail.com'  # Replace with the actual email address

            # Send SMS
            message_sid = send_sms(message_body, twilio_phone_number, recipient_phone_number)
            st.text(f"Message sent with SID: {message_sid}")

            # Send Email
            email_subject = "Accident Detection Alert"
            send_email(email_subject, message_body, recipient_email)
            st.text("Accident detected! Notifications sent to email.")
        else:
            st.text("No accident detected.")
