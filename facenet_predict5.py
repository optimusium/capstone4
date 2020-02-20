# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:44:26 2020

@author: boonping
"""

import cv2
import os,sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
from matplotlib import pyplot as plt

'''
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
'''

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
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras import backend
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import IPython
from scipy import ndimage
from scipy.ndimage.interpolation import shift
from numpy import savetxt,loadtxt
#savetxt('data.csv', data, delimiter=',')
#data = loadtxt('data.csv', delimiter=',')
import gc
from skimage.transform import resize

from mtcnn import MTCNN

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def grayplt(img,title=''):
    '''
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(img[:,:,0],cmap='gray',vmin=0,vmax=1)
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=1)
    plt.title(title, fontproperties=prop)
    '''
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    


    # Show the image
    if np.size(img.shape) == 3:
        ax.imshow(img[:,:,0],cmap='hot',vmin=0,vmax=1)
    else:
        ax.imshow(img,cmap='hot',vmin=0,vmax=1)
   
    plt.show()

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)    


detector = MTCNN()
#images=[]
#for i in range(1,55,1):
#    images.append("frame%i.jpg" % i)
'''
images = ['frame1.jpg','frame2.jpg','frame3.jpg','frame4.jpg','frame5.jpg','frame6.jpg','frame7.jpg','frame8.jpg','frame9.jpg',\
          'frame10.jpg','frame11.jpg','frame12.jpg','frame13.jpg','frame14.jpg','frame15.jpg','frame16.jpg','frame17.jpg','frame18.jpg','frame19.jpg',\
          'frame20.jpg','frame21.jpg','frame22.jpg','frame23.jpg','frame24.jpg','frame25.jpg','frame26.jpg','frame27.jpg','frame28.jpg','frame29.jpg','frame30.jpg','frame31.jpg','frame32.jpg','frame33.jpg','frame34.jpg','frame35.jpg'\
          ]
'''
#images = ['frame1.jpg','frame2.jpg','frame3.jpg','frame4.jpg','frame5.jpg','frame6.jpg','frame7.jpg','frame8.jpg','frame9.jpg','frame10.jpg']
#images = ['frame37.jpg','frame38.jpg','frame39.jpg','frame40.jpg','frame41.jpg','frame42.jpg','frame43.jpg']
images = ['frame56.jpg','frame57.jpg','frame58.jpg','frame59.jpg','frame60.jpg','frame61.jpg']
path=".\\img\\"
path2=".\\img2\\"


model = load_model('facenet/facenet_keras.h5')
model.summary()
print(model.inputs)
print(model.outputs)

model.load_weights("facenet/facenet_keras_weights.h5")

#os.popen("del imgall*")

def image_process(images,gamma,ratioo,flip,blur):    
    img_pointer=0
    for img in images:
        
        imag=cv2.imread(path+img)
        grayplt(imag/255)
        if flip==1:
            imag=np.fliplr(imag)
        if blur==1:
            imag=cv2.GaussianBlur(imag,(5,5),0)
        imag=adjust_gamma(imag,gamma)
        grayplt(imag/255)
        result = detector.detect_faces(imag)
        print("rawimage",result)
        if result==[]: continue
    
        keypoints = result[0]['keypoints']
        
        turned=0
        while keypoints['right_eye'][1]-keypoints['left_eye'][1]>8:
            imag2 = ndimage.rotate(imag, 2, mode='nearest')
            print("turned")
            turned=1
            result2 = detector.detect_faces(imag2)
            if result2==[]: break
            imag=imag2
            result=result2
            keypoints = result[0]['keypoints']
        while keypoints['left_eye'][1]-keypoints['right_eye'][1]>8:
            imag2 = ndimage.rotate(imag, -2, mode='nearest')
            print("turned")
            turned=1
            result2 = detector.detect_faces(imag2)
            if result2==[]: break
            imag=imag2
            result=result2
            keypoints = result[0]['keypoints']
        if turned==1:
            grayplt(imag/255)
    
        # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
        bounding_box = result[0]['box']
        print("bounding_box",bounding_box)
        if bounding_box[3]<45: continue
        if bounding_box[2]<45: continue
    
        keypoints = result[0]['keypoints']
        print("keypoints",keypoints)   
        
        left_bound=int( bounding_box[0]+(keypoints['left_eye'][0]-bounding_box[0])/3 )
        right_bound=int( bounding_box[0]+bounding_box[2]-(bounding_box[0]+bounding_box[2]-keypoints['right_eye'][0])/3 )
        top_bound=int( bounding_box[1]+(min(keypoints['right_eye'][1],keypoints['left_eye'][1])-bounding_box[1])/3 )
        bottom_bound=int( bounding_box[1]+bounding_box[3]-(bounding_box[1]+bounding_box[3]-max(keypoints['mouth_right'][1],keypoints['mouth_left'][1]))/3 )
        
        '''
        left_length=keypoints['nose'][0]-bounding_box[0]
        right_length=bounding_box[2]-keypoints['nose'][0]
        top_length=keypoints['nose'][1]-bounding_box[1]
        bottom_length=bounding_box[3]-keypoints['nose'][1]
        imag=imag[ bounding_box[1]:bounding_box[1]+bounding_box[3] , bounding_box[0]:bounding_box[0]+bounding_box[2] ]
        '''
        left_length=keypoints['nose'][0]-left_bound
        right_length=right_bound-keypoints['nose'][0]
        top_length=keypoints['nose'][1]-top_bound
        bottom_length=bottom_bound-keypoints['nose'][1]        
        imag=imag[top_bound:bottom_bound, left_bound:right_bound]
        #print("32")
        #grayplt(imag/255)
        if left_length>right_length:
            delta=int( (left_length-right_length)/4 )
            res1=np.zeros((imag.shape[0],delta,3))
            imag=np.concatenate((res1,imag,res1),axis=1)
        if right_length>left_length:
            delta=int( (right_length-left_length)/4 )
            res1=np.zeros((imag.shape[0],delta,3))
            imag=np.concatenate((res1,imag,res1),axis=1)
        if top_length>bottom_length:
            delta=int( (top_length-bottom_length)/4 )
            res1=np.zeros((delta,imag.shape[1],3))
            imag=np.concatenate((res1,imag,res1))
        if bottom_length>top_length:
            delta=int( (bottom_length-top_length)/4 )
            res1=np.zeros((delta,imag.shape[1],3))
            imag=np.concatenate((res1,imag,res1))
            
        if imag.shape[0]>=imag.shape[1]:
            ratio=160/imag.shape[0]
            imag = cv2.resize(imag,(int(imag.shape[1]*ratio),160), interpolation = cv2.INTER_CUBIC)
            if imag.shape[1]<160:
                delta=int( (160-imag.shape[1])/2 )
                res1=np.zeros((imag.shape[0],delta,3))
                imag=np.concatenate((res1,imag,res1),axis=1)
            #grayplt(imag/255)
        elif imag.shape[0]<imag.shape[1]:
            ratio=160/imag.shape[1]
            imag = cv2.resize(imag,(160,int(imag.shape[0]*ratio)), interpolation = cv2.INTER_CUBIC)
            if imag.shape[1]<160:
                delta=int( (160-imag.shape[0])/2 )
                res1=np.zeros((delta,imag.shape[1],3))
                imag=np.concatenate((res1,imag,res1),axis=0)
                
            #grayplt(imag/255)
        grayplt(imag/255)
        #continue
            
        img_pointer=eval(img.replace(".jpg","").replace("frame",""))
        for rotate in [-10,0,10]:
            if rotate!=0:
                imag1 = ndimage.rotate(imag, rotate, mode='nearest')
            else:
                imag1=imag
                
            
            for ratioo in [0.8,0.9,1.0,1.1,1.2,1.4,1.5,1.6,1.8]:
            #for ratioo in [0.8]:
                if ratioo==1:
                    imag2 = cv2.resize(imag1,(160, 160), interpolation = cv2.INTER_CUBIC)
                else:
                    imag2 = cv2.resize(imag1,(int(160*ratioo), int(160*ratioo)), interpolation = cv2.INTER_CUBIC)
                #print("before")    
                #grayplt(imag2/255)
                result3a = detector.detect_faces(imag2)
                print("just raw scaled",result3a)
                if result3a==[]: continue
                bounding_box = result3a[0]['box']
                print("bounding_box2: ",bounding_box)
                if bounding_box[3]<45: continue
                if bounding_box[2]<45: continue
        
                keypoints3a = result3a[0]['keypoints']
                coor=keypoints3a["nose"]
                
                if ratioo>1:
                    #print(int(160*(ratioo-1)/2))
                    #imag2 = imag2[int(160*(ratioo-1)/2):160+int(160*(ratioo-1)/2), int(160*(ratioo-1)/2):160+int(160*(ratioo-1)/2)]
                    if coor[1]>=80 and coor[0]>=80:
                        #print("928",coor)
                        imag2=imag2[coor[1]-80:coor[1]+80,coor[0]-80:coor[0]+80]
                    elif coor[1]<80 and coor[0]>=80:
                        #print("923",coor)
                        diff1=80-coor[1]
                        res1=np.zeros((diff1,imag2.shape[1],3))
                        #grayplt(imag2/255)
                        imag2=np.concatenate((res1,imag2))   
                        #grayplt(imag2/255)
                        imag2=imag2[0:160,coor[0]-80:coor[0]+80]
                    elif coor[0]<80 and coor[1]>=80:
                        #print("926",coor)
                        diff1=80-coor[0]
                        res1=np.zeros((imag2.shape[0],diff1,3))
                        imag2=np.concatenate((res1,imag2),axis=1)                        
                        imag2=imag2[coor[1]-80:coor[1]+80,0:160]
                    elif coor[0]<80 and coor[1]<80:
                        #print("927",coor)
                        diff1=80-coor[0]
                        res1=np.zeros((imag2.shape[0],diff1,3))
                        imag2=np.concatenate((res1,imag2),axis=1)                        
                        diff1=80-coor[1]
                        res1=np.zeros((diff1,imag2.shape[1],3))
                        imag2=np.concatenate((res1,imag2))    
                        imag2=imag2[0:160,0:160]
                    if imag2.shape[1]<160:
                        #print("i m here2")
                        diff1=160-imag2.shape[1]
                        res1=np.zeros((imag2.shape[0],diff1,3))
                        imag2=np.concatenate((imag2,res1),axis=1)
                    if imag2.shape[0]<160:
                        diff1=160-imag2.shape[0]
                        res1=np.zeros((diff1,imag2.shape[1],3))
                        imag2=np.concatenate((imag2,res1))
                        
                        
                else:
                    '''
                    diff=160-int(160*ratioo)
                    diff=int(diff/2)
                    res1=np.zeros((diff,imag2.shape[1],3))
                    imag2=np.concatenate((res1,imag2,res1))
                    res1=np.zeros((imag2.shape[0],diff,3))
                    imag2=np.concatenate((res1,imag2,res1),axis=1)
                    '''
                    if coor[0]<80:
                        #print(":i m here")
                        diff1=80-coor[0]
                        res1=np.zeros((imag2.shape[0],diff1,3))
                        imag2=np.concatenate((res1,imag2),axis=1)
                        #grayplt(imag2/255)
                        if imag2.shape[1]<160:
                            #print("i m here2")
                            diff1=160-imag2.shape[1]
                            res1=np.zeros((imag2.shape[0],diff1,3))
                            imag2=np.concatenate((imag2,res1),axis=1)
                            #grayplt(imag2/255)
                        elif imag2.shape[1]>160:
                            imag2=imag2[:,0:160]
                            
                    elif coor[0]>=80:
                        #print(coor[0])
                        diff1=coor[0]-80
                        grayplt(imag2/255)
                        imag2=imag2[:,diff1:160+diff1]
                        grayplt(imag2/255)
                        if imag2.shape[1]<160:
                            print("i m here2")
                            diff1=160-imag2.shape[1]
                            res1=np.zeros((imag2.shape[0],diff1,3))
                            imag2=np.concatenate((imag2,res1),axis=1)
                            grayplt(imag2/255)
                        
                        #res1=np.zeros((imag2.shape[0],diff1,3))
                        #imag2=np.concatenate((res1,imag2,res1),axis=1)
                        
                    #print("intermidat")
                    #grayplt(imag2/255)
                        
                    if coor[1]<80:
                        diff1=80-coor[1]
                        res1=np.zeros((diff1,imag2.shape[1],3))
                        imag2=np.concatenate((res1,imag2))
                        if imag2.shape[0]<160:
                            diff1=160-imag2.shape[0]
                            res1=np.zeros((diff1,imag2.shape[1],3))
                            imag2=np.concatenate((imag2,res1))
                        elif imag2.shape[0]>160:
                            imag2=imag2[0:160,:]
                            
                    elif coor[1]>=80:
                        diff1=coor[1]-80
                        imag2=imag2[diff1:160+diff1,:]
                        if imag2.shape[0]<160:
                            diff1=160-imag2.shape[0]
                            res1=np.zeros((diff1,imag2.shape[1],3))
                            imag2=np.concatenate((imag2,res1))
                        #res1=np.zeros((imag2.shape[0],diff1,3))
                        #imag2=np.concatenate((res1,imag2,res1),axis=1)
                        
                    
                    '''
                    diff2=80-coor[0]
                    res1=np.zeros((diff2,imag2.shape[1],3))
                    imag2=np.concatenate((res1,imag2,res1))
                    '''
                    
                #print("after")
                #grayplt(imag2/255)
                result3 = detector.detect_faces(imag2)
                print("after scale",result3)
                if result3==[]: continue
        
                keypoints3 = result3[0]['keypoints']
    
                if keypoints3=={}:continue
                if 'left_eye' not in keypoints3: continue
                if 'right_eye' not in keypoints3: continue
                if 'mouth_left' not in keypoints3: continue
                if 'mouth_right' not in keypoints3: continue
                if 'nose' not in keypoints3: continue
                bounding_box = result3[0]['box']
                print("bounding_box3: ",bounding_box)
                if bounding_box[3]<45: continue
                if bounding_box[2]<45: continue
            
    
                #print(ratioo)
                #grayplt(imag2/255)
                #continue
                #if ratioo==1.6: 
                #    print(ratioo)
                #    grayplt(imag2/255)

                for sh in [-5,0,5]:
                    if sh!=0:
                        res1 = np.roll(imag2, sh, axis=0)
                        if sh<0:
                            res1[160+sh:160,0:160]=0
                        if sh>0:
                            res1[0:sh,0:160]=0
                    else:
                        res1=imag2
                    
                    for sh2 in [-5,0,5]:
                        
                        if sh2!=0:
                            res2 = np.roll(res1, sh2, axis=1)
                            if sh2<0:
                                res2[0:160,160+sh2:160]=0
                            if sh2>0:
                                res2[0:160,0:sh2]=0
                        else:
                            res2=res1
                        
                        result3 = detector.detect_faces(res2)
                        print("before sample",result3)
                        if result3==[]: continue
                        #if result3=={}: continue
                
                        keypoints3 = result3[0]['keypoints']
            
                        if keypoints3=={}:continue
                        if 'left_eye' not in keypoints3: continue
                        if 'right_eye' not in keypoints3: continue
                        if 'mouth_left' not in keypoints3: continue
                        if 'mouth_right' not in keypoints3: continue
                        if 'nose' not in keypoints3: continue
                        bounding_box = result3[0]['box']
                        print("bounding_box4: ",bounding_box)
                        if bounding_box[3]<45: continue
                        if bounding_box[2]<45: continue
                        if result3[0]['confidence']<0.95: continue
                        
                        print("sampled ",img_pointer,rotate,ratioo,sh,sh2)
                        grayplt(res2/255)
                        res3=np.expand_dims(res2,axis=0)
                        predicted=model.predict(res3)
                        #print(predicted)
                        
                        with open('imgall_merged_representation.csv', "ab") as ff:
                            savetxt(ff, predicted, delimiter=',')
                        with open('imgall_merged_representation_result.csv', "ab") as ff:
                            savetxt(ff, [img_pointer], delimiter=',')
                        
                        
                        
                #raise

                    
        #grayplt(imag/255)
        #raise
        
    
image_process(images,1.0,0.8,0,0)
image_process(images,1.2,0.8,0,0)
image_process(images,0.8,0.8,0,0)

image_process(images,1.0,0.8,1,0)
image_process(images,1.2,0.8,1,0)
image_process(images,0.8,0.8,1,0)

image_process(images,1.0,0.8,0,1)
image_process(images,1.2,0.8,0,1)
image_process(images,0.8,0.8,0,1)

#image_process(images,1.5)
#image_process(images,1.2)
#image_process(images,0.8)