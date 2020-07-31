import cv2

import sys

import logging as log

import datetime as dt

from time import sleep
from time import time



import numpy as np

from matplotlib import pyplot as plt

import multiprocessing


debug=0

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
import threading
import queue

import os

from mtcnn import MTCNN


def grayplt(img,title=''):
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')

    if np.size(img.shape) == 3:
        #ax.imshow(img[:,:,0],cmap='hot',vmin=0,vmax=1)
        ax.imshow(img,vmin=0,vmax=1)
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

def preprocess_image(img):
    imag=cv2.imread(img)
    res = cv2.resize(imag,(160, 160), interpolation = cv2.INTER_CUBIC)
    res=np.expand_dims(res,axis=0)
    return res



def findCosineDistance(source_representation, test_representation):

    a = np.matmul(np.transpose(source_representation), test_representation)

    b = np.sum(np.multiply(source_representation, source_representation))

    c = np.sum(np.multiply(test_representation, test_representation))

    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

 

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

 

def findEuclideanDistance(source_representation, test_representation):

    euclidean_distance = source_representation - test_representation

    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))

    euclidean_distance = np.sqrt(euclidean_distance)

    #euclidean_distance = l2_normalize(euclidean_distance )

    return euclidean_distance



def image_process2(detector,imag,gamma):
    if debug==1: grayplt(imag/255)
    imag=adjust_gamma(imag,gamma)
    if debug==1: grayplt(imag/255)
    result = detector.detect_faces(imag)
    if debug==1: log.deug("rawimage {}".format(result))
    if result==[]: return False,imag
    keypoints = result[0]['keypoints']
    turned=0

    while keypoints['right_eye'][1]-keypoints['left_eye'][1]>8:
        imag2 = ndimage.rotate(imag, 2, mode='nearest')
        if debug==1: log.deug("turned")
        turned=1
        result2 = detector.detect_faces(imag2)
        if result2==[]: break
        imag=imag2
        result=result2
        keypoints = result[0]['keypoints']

    while keypoints['left_eye'][1]-keypoints['right_eye'][1]>8:
        imag2 = ndimage.rotate(imag, -2, mode='nearest')
        if debug==1: log.deug("turned")
        turned=1
        result2 = detector.detect_faces(imag2)
        if result2==[]: break
        imag=imag2
        result=result2
        keypoints = result[0]['keypoints']

    if turned==1:
        if debug==1: grayplt(imag/255)

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.

    bounding_box = result[0]['box']

    if debug==1: log.deug("bounding_box {}".format(bounding_box))
    if bounding_box[3]<45: return False,imag
    if bounding_box[2]<45: return False, imag

    if debug==1: log.deug("keypoints {}".format(keypoints))  

    if keypoints=={}:return False,imag

    if 'left_eye' not in keypoints: return False,imag
    if 'right_eye' not in keypoints: return False,imag
    if 'mouth_left' not in keypoints: return False,imag
    if 'mouth_right' not in keypoints: return False,imag
    if 'nose' not in keypoints: return False,imag

    if result[0]['confidence']<0.95: return False,imag

    left_bound=int( bounding_box[0]) #+(keypoints['left_eye'][0]-bounding_box[0])/3 )
    right_bound=int( bounding_box[0]+bounding_box[2]) #-(bounding_box[0]+bounding_box[2]-keypoints['right_eye'][0])/3 )
    top_bound=int( bounding_box[1]) #+(min(keypoints['right_eye'][1],keypoints['left_eye'][1])-bounding_box[1])/3 )
    bottom_bound=int( bounding_box[1]+bounding_box[3]) #-(bounding_box[1]+bounding_box[3]-max(keypoints['mouth_right'][1],keypoints['mouth_left'][1]))/3 )

    left_length=keypoints['nose'][0]-left_bound
    right_length=right_bound-keypoints['nose'][0]
    top_length=keypoints['nose'][1]-top_bound
    bottom_length=bottom_bound-keypoints['nose'][1]        
    imag=imag[top_bound:bottom_bound, left_bound:right_bound]

    if debug==1: grayplt(imag/255)

    imag=(imag-imag%16)
        #continue
    return True,imag

class FaceRecognizer(object):
    
    def __init__(self):
        self.detector = MTCNN()
        self.api_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.api_path, "model")
        self.root_path = os.path.dirname(self.api_path)
        self.log_path = os.path.join(self.root_path, "facerecognizer.log")
        log.basicConfig(level=log.DEBUG,
                    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                    handlers=[ 
                              log.FileHandler(self.log_path), 
                              log.StreamHandler()
                    ])
        self.__init_model()
        
    def __init_model(self):
        self.model2=load_model(f"{self.model_path}\\facenet_network_model.hdf5")
        self.model2.summary()
        modelname="facenet_network"
        self.model2.load_weights(f"{self.model_path}\\{modelname}.hdf5")
        
        filename=f"{self.model_path}\\MLP0.sav"
        self.model3=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\MLP1.sav"
        self.model4=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\MLP2.sav"
        self.model5=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\MLP3.sav"
        self.model6=pickle.load(open(filename,'rb'))
        
        
        filename=f"{self.model_path}\\LR0.sav"
        self.model7=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\LR1.sav"
        self.model8=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\LR2.sav"
        self.model9=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\LR3.sav"
        self.model10=pickle.load(open(filename,'rb'))
        
        
        filename=f"{self.model_path}\\KNN0.sav"
        self.model11=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\KNN1.sav"
        self.model12=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\KNN2.sav"
        self.model13=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\KNN3.sav"
        self.model14=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\voting0.sav"
        self.model15=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\voting1.sav"
        self.model16=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\voting2.sav"
        self.model17=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\voting3.sav"
        self.model18=pickle.load(open(filename,'rb'))
        
        
        filename=f"{self.model_path}\\svm0.sav"
        self.model19=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\svm1.sav"
        self.model20=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\svm2.sav"
        self.model21=pickle.load(open(filename,'rb'))
        
        filename=f"{self.model_path}\\svm3.sav"
        self.model22=pickle.load(open(filename,'rb'))
        
        
        cascPath = f"{self.model_path}\\haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    ###
    # Perform face recognition and retrieve list of recognized and 
    # unknown face name and images.
    ###
    def perform_face_recognition(self, frame):
        """
        Parameters:
             frame: Open CV captured frame image or video file frame image.
        
        Return:
            list of face name, face location (x, y, w, h) and face image; 
            Unrecognized face name is “unknown”.
            E.g. 
            [  [“unkown”, [10,10,200,200], unknown_face_image],
               [“people1_name”, [15,15,200,200], people1_face_image],
               [“people2_name”, [25,25,200,200], people2_face_image] ] 
        """
    
        start_time = time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
    
        detected_face_count = len(faces)
        log.info(f"detectMultiScale faces: {detected_face_count}")
        
        if detected_face_count == 0:
            return []
        
        face_list = []
        face_count = 0
        
        for (x, y, w, h) in faces:
            face_count += 1
            log.debug(f"face index: {face_count}")
            
            face_detect_time = time()
            check_time = face_detect_time
            gamma=1
            success,imag=image_process2(self.detector, cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2RGB),gamma)  
            sepnt_time = time() - check_time
            log.debug("image_process2 time: {}".format(sepnt_time))
    
            if not success: 
                log.debug("image_process2 failed")
                continue
        
            imag = cv2.resize(imag, (0,0), fx=0.75, fy=0.75) 
   
            check_time = time()
            face_locations = face_recognition.face_locations(imag, number_of_times_to_upsample=1, model='hog') # For GPU, use model='cnn'
            sepnt_time = time() - check_time
            log.debug("face_locations time: {}".format(sepnt_time))
    
            check_time = time()
            face_encodings = face_recognition.face_encodings(imag, face_locations, num_jitters=1)
            sepnt_time = time() - check_time
            log.debug("face_encodings time: {}".format(sepnt_time))
    
            if face_encodings==[]: 
                log.debug("image_process2 failed")
                continue             
    
            check_time = time()
            img2_representation=np.expand_dims(face_encodings[0],axis=0)
            sepnt_time = time() - check_time
            log.debug("expand_dims time: {}".format(sepnt_time))
            
            check_time = time()
            result14=self.model15.predict(img2_representation)
            result15=self.model16.predict(img2_representation)
            result16=self.model17.predict(img2_representation)
            result17=self.model18.predict(img2_representation)
            sepnt_time = time() - check_time
            log.debug("voting time: {}".format(sepnt_time))

            name="unkown"
    
            if result14[0]==1 and result15[0]==0  and result16[0]==0: # and model11.predict_proba(img2_representation)[1]>0.666:
                '''
                if min(model11.kneighbors(img2_representation,return_distance=True)[0][0])>3.0:
                    if debug==1: log.debug("Not sure it is Francis")
                else:
                    if debug==1: log.debug("Francis")
                    name="Francis"
                    '''
                name="Francis"
    
            elif result15[0]==1 and result14[0]==0  and result16[0]==0: # and model12.predict_proba(img2_representation)[1]>0.666:            
                '''
                if min(model11.kneighbors(img2_representation,return_distance=True)[0][0])>3.0:                    
                    if debug==1: log.debug("Not sure it is Yu Ka")
                else:
                    if debug==1: log.debug("Yu Ka")
                    name="Yu Ka"
                '''
                name="Yu Ka"
    
            elif result16[0]==1 and result15[0]==0  and result14[0]==0: # and model13.predict_proba(img2_representation)[1]>0.666:
                '''
                if min(model11.kneighbors(img2_representation,return_distance=True)[0][0])>3.0:
                    if debug==1: log.debug("Not sure it is Boon Ping")
                else:
                    if debug==1: log.debug("Boon Ping")
                    name="Boon Ping"
                '''
                name="Boon Ping"
   
            else:
                name="unknown"
                
                if debug==1: log.debug("Not recognized")
            
            saved_frame=(x,y,w,h)
            face_image = frame[y:y+h, x:x+w]
                
            face_list.append([name, saved_frame, face_image])
    
            sepnt_time = time() - face_detect_time
            log.debug("analyze time: {}".format(sepnt_time))
            
        total_sepnt_time = time() - start_time
        log.debug("total analyze time: {}".format(total_sepnt_time))
    
        return face_list


