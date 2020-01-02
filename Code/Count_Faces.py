# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 22:48:12 2018

@author: Sonu
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2 
import os
from sklearn.model_selection import train_test_split
import keras
from custom_metric import fmeasure
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Input
from keras.models import Model 
from keras import backend as K
import itertools
from keras.utils import to_categorical

def read_Images_coordinates_from_csv(filename):
    df= pd.read_csv(filename,nrows=10)
    d=defaultdict(list)
    
    
    for i in range(0,len(df)):
        bb1=df.loc[i,'xmin']
        bb2=df.loc[i,'xmax']
        bb3=df.loc[i,'ymin']
        bb4=df.loc[i,'ymax']
        bbox=[bb1,bb2,bb3,bb4]
        name=df.loc[i,'Name']
        d[name].append(bbox)
        
    imgpath = np.empty(len(d), dtype=object)
    images =  np.empty(len(d), dtype=object)
    n=0
    head_count=list()
    
    for key,value in d.items():
        head_count.append(len(value))
    #headcount=pd.DataFrame(head_count)        
    
    #Mapping the image name to the actual image
    for key,value in d.items():
        imgpath[n] = os.path.join(r"C:\Users\Sonu\Documents\Carrear\HACKATHONS\Computer Vision\Counting Faces\image_data",key)
        images[n] = cv2.imread(imgpath[n])
        #cv2.imshow('original',images[n])
        #cv2.waitKey(2000) 
        #cv2.destroyAllWindows()
        #images[n] = cv2.resize(images[n], (612, 408))
        images[n] = images[n].astype('float32')
        images[n] /= 255  
        n=n+1
    return d,images,head_count
    
dictionary,image,headcount=read_Images_coordinates_from_csv(r'C:\Users\Sonu\Documents\Carrear\HACKATHONS\Computer Vision\Counting Faces\bbox_train.csv')

#Function to find the total head count in each image
def find_head_count(dictionary):
    head_count=[]
    for key,value in dictionary.items():
        head_count.append(value)
    return head_count    

label=find_head_count(dictionary)

#Appending Images to list
data=[]
for img in image:
    data.append(img)

#Spliting of datset into tarin and test set
image_train, image_validation, label_train, label_validation,coordinate_train,coordinate_validation = train_test_split(data, headcount, label, test_size=0.2, random_state=42)

#Function to extract each bounding box Coordinate in an image.
def Extract_each_bounding_box_coordinate_from_image(image,coordinate,label):
    i=0
    for j in range(len(image)):
        cv2.imshow('original',image[j])
        cv2.waitKey(2000) 
        cv2.destroyAllWindows() 
        print('coor',coordinate[j])
        print('label',label[j])
        k=[x for x in itertools.chain.from_iterable(coordinate[j])]
        while i<len(k):
            print(k[i:i+4])
            i=i+4
        i=0    
    return image,coordinate,label   

image_train,coordinate_train,label_train=Extract_each_bounding_box_coordinate_from_image(image_train,coordinate_train,label_train)
image_validation,coordinate_validation,label_validation=Extract_each_bounding_box_coordinate_from_image(image_validation,coordinate_validation,label_validation)  

#Function to map each bbox with its corresponding labels and images.
def mapping_individual_bbox(image,label,coordinate):    
    i=0
    l=list()
    m=list()
    n=list()
    for j in range(len(image)):
        label=label[j]
        k=[x for x in itertools.chain.from_iterable(coordinate[j])]
        while i<len(k):
            coor = k[i:i+4]
            i=i+4
            l.append(coor)
            m.append(label)
            n.append(image[j])
        i=0      
    return l,m,n   
         
coordinate_train,label_train,image_train=mapping_individual_bbox(image_train,coordinate_train,label_train)      
coordinate_validation,label_validation,image_validation=mapping_individual_bbox(image_validation,coordinate_validation,coordinate_validation)

#Creating one_hot labels 
one_hot_label_train =  to_categorical(label_train)
one_hot_label_validation =  to_categorical(label_validation)

def distance_loss(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=1)), axis=0) 


def iou_metric(y_true, y_pred):
    # iou as metric for bounding box regression
    # input must be as [x1, y1, x2, y2]
    
    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0] + 1) * K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1)
    
    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] + 1) * K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    overlap_1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection
    
    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    return iou    


batch_size = 32
nb_classes = len(set(label_train))
nb_epoch = 5
img_rows = 612 
img_cols = 408

#Multi output Model Building
input_shape = Input(shape=(img_cols,img_rows,3)) 
conv_0=Conv2D(4, (3, 3), activation='relu')(input_shape)
pool_0=MaxPooling2D(pool_size=(2, 2))(conv_0)
conv_1=Conv2D(8, (3, 3), activation='relu')(pool_0)
pool_1=MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2=Conv2D(16, (3, 3), activation='relu')(pool_1)
pool_2=MaxPooling2D(pool_size=(2, 2))(conv_2)
flat=Flatten()(pool_2)
dense_1=Dense(1024, activation='relu')(flat)
drop_1=Dropout(0.05)(dense_1)
dense_2=Dense(512, activation='relu')(drop_1)
drop_2=Dropout(0.1)(dense_2)
dense_3=Dense(256, activation='relu')(drop_2)
drop_3=Dropout(0.15)(dense_3)
dense_4=Dense(128, activation='relu')(drop_3)
drop_4=Dropout(0.2)(dense_4)
dense_5=Dense(64, activation='relu')(drop_4)
drop_5=Dropout(0.25)(dense_5)
out_1=Dense(nb_classes, activation='softmax')(drop_5)
out_2=Dense(4, activation='linear')(drop_5)

model= Model(inputs=input_shape, outputs=[out_1,out_2])

#Model Compilation
model.compile(loss=['categorical_crossentropy',distance_loss],optimizer=keras.optimizers.Adam(lr = 0.001),metrics={'categorical_crossentropy':fmeasure, distance_loss:iou_metric})
checkpoint=keras.callbacks.ModelCheckpoint(filepath='./Counting_Faces.h5',monitor='val_loss', save_best_only=True, mode=min, verbose=0)

#Model Training
history=model.fit(np.array(image_train), [np.array(one_hot_label_train),np.array(coordinate_train)], batch_size=10, epochs=nb_epoch, validation_data=(np.array(image_validation),[np.array(one_hot_label_validation),np.array(coordinate_validation)]), callbacks=[checkpoint])

#Model Evaluation on validation data
score = model.evaluate(np.array(image_validation),[np.array(one_hot_label_validation),np.array(coordinate_validation)], batch_size=10, verbose=1)