# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:01:23 2020

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
#from sklearn.metrics import multilabel_confusion_matrix

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


import process_csv
'''
X0 = loadtxt('img0_merged_representation.csv', delimiter=',')
X1 = loadtxt('img1_merged_representation.csv', delimiter=',')
X2 = loadtxt('img2_merged_representation.csv', delimiter=',')
X3 = loadtxt('img4_merged_representation.csv', delimiter=',')
X4 = loadtxt('img3_merged_representation.csv', delimiter=',')


X5 = loadtxt('img6_merged_representation.csv', delimiter=',')
X6 = loadtxt('img7_merged_representation.csv', delimiter=',')
X7 = loadtxt('img8_merged_representation.csv', delimiter=',')
X8 = loadtxt('img9_merged_representation.csv', delimiter=',')
X9 = loadtxt('img10_merged_representation.csv', delimiter=',')
X10 = loadtxt('img11_merged_representation.csv', delimiter=',')
X11 = loadtxt('img12_merged_representation.csv', delimiter=',')
X12 = loadtxt('img13_merged_representation.csv', delimiter=',')
X13 = loadtxt('img14_merged_representation.csv', delimiter=',')


Y0=np.append( np.ones(X0.shape[0]), np.zeros(X1.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(X2.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(X3.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(X4.shape[0]),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)
Y0=np.append( Y0, np.zeros(50),axis=0)

Y1=np.append( np.zeros(X0.shape[0]),  np.ones(X1.shape[0]),axis=0)
Y1=np.append( Y1, np.zeros(X2.shape[0]),axis=0)
Y1=np.append( Y1, np.zeros(X3.shape[0]),axis=0)
Y1=np.append( Y1, np.zeros(X4.shape[0]),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)
Y1=np.append( Y1, np.zeros(50),axis=0)

Y2=np.append( np.zeros(X0.shape[0]), np.zeros(X1.shape[0]),axis=0)
Y2=np.append( Y2, np.ones(X2.shape[0]),axis=0)
Y2=np.append( Y2, np.zeros(X3.shape[0]),axis=0)
Y2=np.append( Y2, np.zeros(X4.shape[0]),axis=0)

Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)
Y2=np.append( Y2, np.zeros(50),axis=0)

Y3=np.append( np.zeros(X0.shape[0]), np.zeros(X1.shape[0]),axis=0)
Y3=np.append( Y3, np.zeros(X2.shape[0]),axis=0)
Y3=np.append( Y3, np.ones(X3.shape[0]),axis=0)
Y3=np.append( Y3, np.zeros(X4.shape[0]),axis=0)

Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)
Y3=np.append( Y3, np.zeros(50),axis=0)


X=X0
X=np.append(X,X1,axis=0)
X=np.append(X,X2,axis=0)
X=np.append(X,X3,axis=0)
X=np.append(X,X4,axis=0)

for i in [0,600,1200,1800,2400]:
    X=np.append(X,X5[i:i+10],axis=0)
    X=np.append(X,X6[i:i+10],axis=0)
    X=np.append(X,X7[i:i+10],axis=0)
    X=np.append(X,X8[i:i+10],axis=0)
    X=np.append(X,X9[i:i+10],axis=0)
    X=np.append(X,X10[i:i+10],axis=0)
    X=np.append(X,X11[i:i+10],axis=0)
    X=np.append(X,X12[i:i+10],axis=0)
    X=np.append(X,X13[i:i+10],axis=0)

print(X.shape)
print(Y3.shape)
#raise
X_train,X_test,Y_train0,Y_test0,Y_train1,Y_test1,Y_train2,Y_test2,Y_train3,Y_test3 = train_test_split(X,Y0,Y1,Y2,Y3,test_size = 0.1)
'''
X_train,X_test,Y_train0,Y_test0,Y_train1,Y_test1,Y_train2,Y_test2,Y_train3,Y_test3 = process_csv.process_csv()

print("KNN0")

#KNN = RandomForestClassifier(n_jobs=200,max_depth=None, max_leaf_nodes=5, random_state=0)
KNN=KNeighborsClassifier(n_neighbors=5, weights='distance') #LogisticRegression(random_state=0, C=1.0)
#KNN = BaggingClassifier(base_estimator=KNNa,n_estimators=10, random_state=0)
KNN.fit(X_train,Y_train0)

filename="KNN0.sav"
pickle.dump(KNN,open(filename,'wb'))
KNN=pickle.load(open(filename,'rb'))
prediction=KNN.predict(X_test)
CM=confusion_matrix(Y_test0,prediction)
print(CM)


os.popen("del not0.csv")
os.popen("del not0_result.csv")
import random
extra=np.array([])
for i in range(X_train.shape[0]):
    if i%3==1: continue
    if i%3==2:
        if i%5!=2: continue
    if random.random()<0.2:continue    
    found=0
    tim=0
    while found==0 and tim<3:
        tim+=1
        extr=X_train[i]+np.ones(128)*random.random()*0.7
        extrr=np.expand_dims(extr,axis=0)
        dist=KNN.kneighbors(extrr,return_distance=True)
        prediction=KNN.predict(extrr)
        #print(dist)
        #print(prediction)
    
        if prediction[0]==1 and min(dist[0][0])>4:
            extra=np.append(extra, extr, axis=0)
            found=1
    if i%100==0: print(i,extra.shape)
    #if i==10: break
extra=extra.reshape(int(extra.shape[0]/128),128)
#print(extra.shape)
extra_result=np.zeros(extra.shape[0])
    
os.popen("del KNN0.sav")
with open('not0.csv', "ab") as ff:
    savetxt(ff, extra, delimiter=',')    
with open('not0_result.csv', "ab") as ff:
    savetxt(ff, extra_result, delimiter=',')    

X_expand=np.append(X_train,extra,axis=0)
Y_expand=np.append(Y_train0,extra_result,axis=0)
print("923",X_train.shape)
print(X_expand.shape)
print(Y_expand.shape)


#KNN = RandomForestClassifier(n_jobs=200,max_depth=None, max_leaf_nodes=5, random_state=0)
KNNa=KNeighborsClassifier(n_neighbors=5, weights='distance') #LogisticRegression(random_state=0, C=1.0)
#KNN = BaggingClassifier(base_estimator=KNNa,n_estimators=10, random_state=0)
KNNa.fit(X_expand,Y_expand)

filename="KNN0.sav"
pickle.dump(KNNa,open(filename,'wb'))
KNNa=pickle.load(open(filename,'rb'))
prediction=KNNa.predict(X_test)
CM=confusion_matrix(Y_test0,prediction)
print(CM)


#raise

print("KNN1")
KNN1=KNeighborsClassifier(n_neighbors=5, weights='distance') #LogisticRegression(random_state=0, C=1.0)
#KNN1 = RandomForestClassifier(n_jobs=200,max_depth=None, max_leaf_nodes=5, random_state=5)
#KNN1 = BaggingClassifier(base_estimator=KNN1a,n_estimators=10, random_state=0)
KNN1.fit(X_train,Y_train1)
filename="KNN1.sav"
pickle.dump(KNN1,open(filename,'wb'))
KNN1=pickle.load(open(filename,'rb'))
prediction=KNN1.predict(X_test)
CM=confusion_matrix(Y_test1,prediction)
print(CM)

os.popen("del not1.csv")
os.popen("del not1_result.csv")
import random
extra=np.array([])
for i in range(X_train.shape[0]):
    if i%3==1: continue
    if i%3==2:
        if i%5!=2: continue
    if random.random()<0.2:continue        
    found=0
    tim=0
    while found==0 and tim<3:
        tim+=1
        extr=X_train[i]+np.ones(128)*random.random()*0.7
        extrr=np.expand_dims(extr,axis=0)
        dist=KNN1.kneighbors(extrr,return_distance=True)
        prediction=KNN1.predict(extrr)
        
        #print(dist)
        #print(prediction)
    
        if prediction[0]==1 and min(dist[0][0])>4:
            extra=np.append(extra, extr, axis=0)
            found=1
        #print(extra.shape)
    #if i==10: break
extra=extra.reshape(int(extra.shape[0]/128),128)
#print(extra.shape)
extra_result=np.zeros(extra.shape[0])
    
os.popen("del KNN1.sav")
with open('not1.csv', "ab") as ff:
    savetxt(ff, extra, delimiter=',')    
with open('not1_result.csv', "ab") as ff:
    savetxt(ff, extra_result, delimiter=',')    

X_expand=np.append(X_train,extra,axis=0)
Y_expand=np.append(Y_train1,extra_result,axis=0)
print(X_expand.shape)
print(Y_expand.shape)


#KNN = RandomForestClassifier(n_jobs=200,max_depth=None, max_leaf_nodes=5, random_state=0)
KNN1a=KNeighborsClassifier(n_neighbors=5, weights='distance') #LogisticRegression(random_state=0, C=1.0)
#KNN = BaggingClassifier(base_estimator=KNNa,n_estimators=10, random_state=0)
KNN1a.fit(X_expand,Y_expand)

filename="KNN1.sav"
pickle.dump(KNN1a,open(filename,'wb'))
KNN1a=pickle.load(open(filename,'rb'))
prediction=KNN1a.predict(X_test)
CM=confusion_matrix(Y_test1,prediction)
print(CM)


print("KNN2")
#KNN2 = RandomForestClassifier(n_jobs=200,max_depth=None, max_leaf_nodes=5, random_state=5)
KNN2=KNeighborsClassifier(n_neighbors=5, weights='distance') #LogisticRegression(random_state=0, C=1.0)
#KNN2 = BaggingClassifier(base_estimator=KNN2a,n_estimators=10, random_state=0)

KNN2.fit(X_train,Y_train2)
filename="KNN2.sav"
pickle.dump(KNN2,open(filename,'wb'))
KNN2=pickle.load(open(filename,'rb'))
prediction=KNN2.predict(X_test)
CM=confusion_matrix(Y_test2,prediction)
print(CM)

os.popen("del not2.csv")
os.popen("del not2_result.csv")
import random
extra=np.array([])
for i in range(X_train.shape[0]):
    if i%3==1: continue
    if i%3==2:
        if i%5!=2: continue
    if random.random()<0.2:continue        
    found=0
    tim=0
    while found==0 and tim<3:
        tim+=1
        extr=X_train[i]+np.ones(128)*random.random()*0.7
        extrr=np.expand_dims(extr,axis=0)
        dist=KNN2.kneighbors(extrr,return_distance=True)
        prediction=KNN2.predict(extrr)
        #print(dist)
        #print(prediction)
    
        if prediction[0]==1 and min(dist[0][0])>4:
            extra=np.append(extra, extr, axis=0)
            found=1
        #print(extra.shape)
    #if i==10: break
extra=extra.reshape(int(extra.shape[0]/128),128)
#print(extra.shape)
extra_result=np.zeros(extra.shape[0])
    
os.popen("del KNN2.sav")
with open('not2.csv', "ab") as ff:
    savetxt(ff, extra, delimiter=',')    
with open('not2_result.csv', "ab") as ff:
    savetxt(ff, extra_result, delimiter=',')    

X_expand=np.append(X_train,extra,axis=0)
Y_expand=np.append(Y_train2,extra_result,axis=0)
print(X_expand.shape)
print(Y_expand.shape)


#KNN = RandomForestClassifier(n_jobs=200,max_depth=None, max_leaf_nodes=5, random_state=0)
KNN2a=KNeighborsClassifier(n_neighbors=5, weights='distance') #LogisticRegression(random_state=0, C=1.0)
#KNN = BaggingClassifier(base_estimator=KNNa,n_estimators=10, random_state=0)
KNN2a.fit(X_expand,Y_expand)

filename="KNN2.sav"
pickle.dump(KNN2a,open(filename,'wb'))
KNN2a=pickle.load(open(filename,'rb'))
prediction=KNN2a.predict(X_test)
CM=confusion_matrix(Y_test2,prediction)
print(CM)


print("KNN3")
#
KNN3=KNeighborsClassifier(n_neighbors=5, weights='distance') #LogisticRegression(random_state=0, C=1.0)
#KNN3 = BaggingClassifier(base_estimator=KNN3a,n_estimators=10, random_state=0)
#KNN3 = RandomForestClassifier(n_jobs=200,max_depth=None, max_leaf_nodes=5, random_state=5)
KNN3.fit(X_train,Y_train3)
filename="KNN3.sav"
pickle.dump(KNN3,open(filename,'wb'))
KNN3=pickle.load(open(filename,'rb'))
prediction=KNN3.predict(X_test)
CM=confusion_matrix(Y_test3,prediction)
print(CM)

'''
os.popen("del not3.csv")
os.popen("del not3_result.csv")
import random
extra=np.array([])
for i in range(X_train.shape[0]):
    if i%3==1: continue
    if i%3==2:
        if i%5!=2: continue
    if random.random()<0.2:continue    

    found=0
    tim=0
    while found==0 and tim<3:
        tim+=1
        extr=X_train[i]+np.ones(128)*random.random()*0.7
        extrr=np.expand_dims(extr,axis=0)
        dist=KNN3.kneighbors(extrr,return_distance=True)
        prediction=KNN3.predict(extrr)
        #print(dist)
        #print(prediction)
    
        if prediction[0]==1 and min(dist[0][0])>4:
            extra=np.append(extra, extr, axis=0)
            found=1
        #print(extra.shape)
    #if i==10: break
extra=extra.reshape(int(extra.shape[0]/128),128)
#print(extra.shape)
extra_result=np.zeros(extra.shape[0])
    
os.popen("del KNN3.sav")
with open('not3.csv', "ab") as ff:
    savetxt(ff, extra, delimiter=',')    
with open('not3_result.csv', "ab") as ff:
    savetxt(ff, extra_result, delimiter=',')    

X_expand=np.append(X_train,extra,axis=0)
Y_expand=np.append(Y_train3,extra_result,axis=0)
print(X_expand.shape)
print(Y_expand.shape)


#KNN = RandomForestClassifier(n_jobs=200,max_depth=None, max_leaf_nodes=5, random_state=0)
KNN3a=KNeighborsClassifier(n_neighbors=5, weights='distance') #LogisticRegression(random_state=0, C=1.0)
#KNN = BaggingClassifier(base_estimator=KNNa,n_estimators=10, random_state=0)
KNN3a.fit(X_expand,Y_expand)

filename="KNN3.sav"
pickle.dump(KNN3a,open(filename,'wb'))
KNN3a=pickle.load(open(filename,'rb'))
prediction=KNN3a.predict(X_test)
CM=confusion_matrix(Y_test3,prediction)
print(CM)
'''