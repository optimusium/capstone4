# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:13:21 2020

@author: User
"""
import os,re
import cv2
import sys
import logging as log
import datetime as dt

from time import sleep
from time import time

import numpy as np

from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten,Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import add,Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical,plot_model

from tensorflow.keras import optimizers
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import IPython
from scipy import ndimage
from scipy.ndimage.interpolation import shift
from numpy import savetxt,loadtxt

#savetxt('data.csv', data, delimiter=',')

#data = loadtxt('data.csv', delimiter=',')

import gc
from skimage.transform import resize

import pickle
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

import face_recognition

from mtcnn import MTCNN


def execfile(pyfile):
    with open(pyfile) as f:
        code = compile(f.read(), pyfile, 'exec')
        exec(code)
    
    
def select_operation(cmd):
    if cmd==1: #application
        execfile("webcam_cv3_dlib2.py")
        #raise
    elif cmd==2: #full training
        execfile("facenet_predict6.py") #pre-process all images and encode the images
        execfile("knn_dlib.py") #Running classification
        execfile("logistic_regression_dlib.py")
        execfile("mlp_dlib.py")
        execfile("svm_dlib.py")
        execfile("voting_dlib.py") #voting
        #raise
    elif cmd==3: #capture image
        execfile("webcam_cv3_capture.py") #after capture image, user can review frame.jpg then go to step 4.
    elif cmd==4: #save image
        lookup=open("lookup.csv","r") #this is where the image are tagged with number label
        lines=lookup.readlines()
        buf=0
        last=""
        for line in lines:
            if line.find(",")==-1: continue
            buf=eval( re.split(",",line)[0] )+1
        print(buf)
        lookup.close()
        os.popen("copy frame.jpg img\\frame%i.jpg" % buf)
        lookup2=open("lookup.csv","a+")
        print("Press 1 for Francis")
        print("Press 2 for Yu Ka")
        print("Press 3 for Boonping")
        print("Press 0 for Others")
        Val=input()
        lookup2.write("%i,%s\n"%(buf,Val))
        lookup2.close() #after forming lookup table based on user input, save lookup.csv
        
        
        
        
        
if __name__=="__main__":
    while 1:
        print("Press 1 for Application")
        print("Press 2 for Training")
        print("Press 3 for Image Capturing")
        print("Press 4 for Saving Captured Image")
        inp=eval(input())
        
        select_operation(inp)
