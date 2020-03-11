# -*- coding: utf-8 -*-

"""

Created on Fri Jan 10 23:13:44 2020



@author: boonping

"""

import cv2

import os,sys

import logging as log

import datetime as dt

from time import sleep

import numpy as np

from matplotlib import pyplot as plt

import pickle

'''

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

'''

from sklearn.model_selection import train_test_split



from scipy import ndimage

from scipy.ndimage.interpolation import shift

from numpy import savetxt,loadtxt



from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix

import process_csv_dlib

#from sklearn.metrics import multilabel_confusion_matrix

X_train,X_test,Y_train0,Y_test0,Y_train1,Y_test1,Y_train2,Y_test2,Y_train3,Y_test3 = process_csv_dlib.process_csv()



'''

print("Start PCA")

pca=PCA(n_components=16)

pca.fit(X_train)

filename="pca.sav"

pickle.dump(pca,open(filename,'wb'))

pca=pickle.load(open(filename,'rb'))



print("PCA done")

X_train1=pca.transform(X_train)

X_test1=pca.transform(X_test)

print(X_train1.shape)

'''

X_train1=X_train

X_test1=X_test



'''

print("Start SVM")

model=OneVsRestClassifier( SVC(kernel='linear',probability=True, C=0.5, gamma='auto') )

model.fit(X_train1,Y_train)

print("fit SVM")

filename="svm.sav"

pickle.dump(model,open(filename,'wb'))

model=pickle.load(open(filename,'rb'))

prediction=model.predict(X_test1)

CM=confusion_matrix(Y_test,prediction)

print(CM)

'''





model=SVC(kernel='linear',probability=True, C=0.6, gamma='auto')

model.fit(X_train1,Y_train0)

print("fit SVM0")

filename="svm0.sav"

pickle.dump(model,open(filename,'wb'))

model=pickle.load(open(filename,'rb'))

prediction=model.predict(X_test1)

CM=confusion_matrix(Y_test0,prediction)

print(CM)



model1=SVC(kernel='linear',probability=True, C=0.6, gamma='auto')

model1.fit(X_train1,Y_train1)

print("fit SVM1")

filename="svm1.sav"

pickle.dump(model1,open(filename,'wb'))

model1=pickle.load(open(filename,'rb'))

prediction=model1.predict(X_test1)

CM=confusion_matrix(Y_test1,prediction)

print(CM)

    

model2=SVC(kernel='linear',probability=True, C=0.6, gamma='auto')

model2.fit(X_train1,Y_train2)

print("fit SVM2")

filename="svm2.sav"

pickle.dump(model2,open(filename,'wb'))

model2=pickle.load(open(filename,'rb'))

prediction=model2.predict(X_test1)

CM=confusion_matrix(Y_test2,prediction)

print(CM)



model3=SVC(kernel='linear',probability=True, C=0.6, gamma='auto')

model3.fit(X_train1,Y_train3)

print("fit SVM3")

filename="svm3.sav"

pickle.dump(model3,open(filename,'wb'))

model3=pickle.load(open(filename,'rb'))

prediction=model3.predict(X_test1)

CM=confusion_matrix(Y_test3,prediction)

print(CM)