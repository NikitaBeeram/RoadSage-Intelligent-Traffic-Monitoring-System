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

# Capture video frames
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

X_train = base_model.predict(X_train, verbose=0)
X_valid = base_model.predict(X_valid, verbose=0)
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

# Train and save the model
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid), verbose=0)
model.save("model.h5")
print("Model saved!")
