#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.io
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd

from keras.models import load_model
model = load_model("model.h5")

testx = np.load('testx.npy', allow_pickle=True)

testy = np.load('testy.npy', allow_pickle=True)

plt.imshow(testx[15])
print(testy[15])

score = model.evaluate(testx, testy)

print('Test loss:', score[0]) 
print('Test accuracy:', score[1]*100)

pred = model.predict(testx) 
pred = np.argmax(pred, axis = 1)[:15] 
label = np.argmax(testy,axis = 1)[:15] 

print("Predicted labels:",pred) 
print("Actual Labels:   ",label)