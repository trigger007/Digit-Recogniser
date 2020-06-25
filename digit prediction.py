# -*- coding: utensorflow-8 -*-
"""
Created on Thu Jun 25 13:45:06 2020

@author: ADITYA SINGH
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
import cv2


data=pd.read_csv("train.csv")

y = data["label"]
x = data.drop(labels = ["label"], axis = 1) 

x = x.values.reshape(-1, 28, 28, 1)

y = to_categorical(y)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   rotation_range=10,
                                   zoom_range = 0.2)

train_datagen.fit(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

cnn = Sequential()

cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28, 28, 1]))
cnn.add(MaxPool2D(pool_size=2, strides=2))

cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPool2D(pool_size=2, strides=2))

cnn.add(Flatten())

cnn.add(Dense(units=128, activation='relu'))

cnn.add(Dense(units=10, activation='softmax'))

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


cnn.fit(x_train,y_train, batch_size=32,  epochs = 25)

y_pred=cnn.predict(x_test)
y_pred[y_pred<0.5]=0
y_pred[y_pred>0.7]=1

l=[]

for i in range(len(y_pred)):
    if y_pred[i][0]==1:
        l.append(0)
    elif  y_pred[i][1]==1:
        l.append(1)
    elif  y_pred[i][2]==1:
        l.append(2)
    elif  y_pred[i][3]==1:
        l.append(3)
    elif  y_pred[i][4]==1:
        l.append(4)
    elif  y_pred[i][5]==1:
        l.append(5)
    elif  y_pred[i][6]==1:
        l.append(6)
    elif  y_pred[i][7]==1:
        l.append(7)
    elif  y_pred[i][8]==1:
        l.append(8)
    else:  
        l.append(9)

inp=pd.read_csv("test.csv")
inp = inp.values.reshape(-1, 28, 28, 1)

out=cnn.predict(inp)

out[out<0.5]=0
out[out>0.7]=1

l=[]

for i in range(len(out)):
    if out[i][0]==1:
        l.append(0)
    elif  out[i][1]==1:
        l.append(1)
    elif  out[i][2]==1:
        l.append(2)
    elif  out[i][3]==1:
        l.append(3)
    elif  out[i][4]==1:
        l.append(4)
    elif  out[i][5]==1:
        l.append(5)
    elif  out[i][6]==1:
        l.append(6)
    elif  out[i][7]==1:
        l.append(7)
    elif  out[i][8]==1:
        l.append(8)
    else:  
        l.append(9)

l1=[]
for i in range(1,len(l)+1):
    l1.append(i)

dfo = pd.DataFrame({'ImageId':l1, 'Label':l})

dfo.to_csv('imagesubmission.csv')





    


