import cv2

import sys
import logging as log
import datetime as dt

from time import sleep
from time import time

import numpy as np
from matplotlib import pyplot as plt

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

from mtcnn import MTCNN
detector = MTCNN()

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def grayplt(img,title=''):
  
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')

    if np.size(img.shape) == 3:
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

def image_process(imag,gamma):    
    img_pointer=0
    if 1:
        if debug==1: grayplt(imag/255)
        imag=adjust_gamma(imag,gamma)
        if debug==1: grayplt(imag/255)
        result = detector.detect_faces(imag)
        if debug==1: print("rawimage",result)
        if result==[]: return False,imag
        keypoints = result[0]['keypoints']        

        turned=0
        while keypoints['right_eye'][1]-keypoints['left_eye'][1]>8:
            imag2 = ndimage.rotate(imag, 2, mode='nearest')
            if debug==1: print("turned")
            turned=1
            result2 = detector.detect_faces(imag2)
            if result2==[]: break
            imag=imag2
            result=result2
            keypoints = result[0]['keypoints']

        while keypoints['left_eye'][1]-keypoints['right_eye'][1]>8:
            imag2 = ndimage.rotate(imag, -2, mode='nearest')
            if debug==1: print("turned")
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
        if debug==1: print("bounding_box",bounding_box)
        if bounding_box[3]<45: return False,imag
        if bounding_box[2]<45: return False, imag
        keypoints = result[0]['keypoints']
        if debug==1: print("keypoints",keypoints)      

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
    return True,imag



'''

#import tensorflow as tf

model = load_model('facenet/facenet_keras.h5')

model.summary()

if debug==1: print(model.inputs)

if debug==1: print(model.outputs)



model.load_weights("facenet/facenet_keras_weights.h5")

'''

'''

def createModel():

    

    inputShape=(128,)

    inputs      = Input(shape=inputShape)

    #x=Reshape((128,1))(inputs)

    x = Dense(128,activation="relu")(inputs)

    #x=Conv1D(128,kernel_size=(8,),activation="relu",padding="same")(x)

    #x=Conv1D(64,kernel_size=(4,),activation="relu",padding="same")(x)

    #x=AveragePooling1D(4)(x)

    #x = Flatten()(x)

    x = Dense(64,activation="relu")(x)

    x = Dense(32,activation="relu")(x)

    x = Dense(20,activation="relu")(x)



    outputs0 = Dense(20,activation="relu")(x)

    outputs1 = Dense(20,activation="relu")(x)

    outputs2 = Dense(20,activation="relu")(x)

    outputs3 = Dense(20,activation="relu")(x)

    

    outputs0 = Dense(2,activation="softmax")(outputs0)

    outputs1 = Dense(2,activation="softmax")(outputs1)

    outputs2 = Dense(2,activation="softmax")(outputs2)

    outputs3 = Dense(2,activation="softmax")(outputs3)

    

    model       = Model(inputs=inputs,outputs=[outputs0,outputs1,outputs2,outputs3])       

    #model       = Model(inputs=[inputs0,inputs1,inputs2,inputs3,inputs4,inputs5,inputs6,inputs7],outputs=outputs)       

    model.compile(loss='categorical_crossentropy', 

                optimizer=optimizers.Adam() ,

                metrics=['accuracy'])



    return model

'''



#model2=createModel()

model2=load_model('facenet_network_model.hdf5')

model2.summary()

modelname="facenet_network"

model2.load_weights(modelname + ".hdf5")



'''

filename="svm0.sav"

model3=pickle.load(open(filename,'rb'))

filename="svm1.sav"

model4=pickle.load(open(filename,'rb'))

filename="svm2.sav"

model5=pickle.load(open(filename,'rb'))

filename="svm3.sav"

model6=pickle.load(open(filename,'rb'))

'''

filename="MLP0.sav"

model3=pickle.load(open(filename,'rb'))

filename="MLP1.sav"

model4=pickle.load(open(filename,'rb'))

filename="MLP2.sav"

model5=pickle.load(open(filename,'rb'))

filename="MLP3.sav"

model6=pickle.load(open(filename,'rb'))



filename="LR0.sav"

model7=pickle.load(open(filename,'rb'))

filename="LR1.sav"

model8=pickle.load(open(filename,'rb'))

filename="LR2.sav"

model9=pickle.load(open(filename,'rb'))

filename="LR3.sav"

model10=pickle.load(open(filename,'rb'))



filename="KNN0.sav"

model11=pickle.load(open(filename,'rb'))

filename="KNN1.sav"

model12=pickle.load(open(filename,'rb'))

filename="KNN2.sav"

model13=pickle.load(open(filename,'rb'))

filename="KNN3.sav"

model14=pickle.load(open(filename,'rb'))



filename="voting0.sav"

model15=pickle.load(open(filename,'rb'))

filename="voting1.sav"

model16=pickle.load(open(filename,'rb'))

filename="voting2.sav"

model17=pickle.load(open(filename,'rb'))

filename="voting3.sav"

model18=pickle.load(open(filename,'rb'))



filename="svm0.sav"

model19=pickle.load(open(filename,'rb'))

filename="svm1.sav"

model20=pickle.load(open(filename,'rb'))

filename="svm2.sav"

model21=pickle.load(open(filename,'rb'))

filename="svm3.sav"

model22=pickle.load(open(filename,'rb'))



cascPath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

log.basicConfig(filename='webcam.log',level=log.INFO)



video_capture = cv2.VideoCapture(0)

anterior = 0



numm=-1



prev=0

changed=1

static_count=0
frame_time=0

while True:

    if not video_capture.isOpened():
        if debug==1: print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #cv2.imwrite("frame.jpg",frame)
    #sleep(0.25)

    if changed==1:
        cv2.imshow('Video', frame)
        changed=0
    time1=time()


    prev_frame=frame

    #if debug==1: print( np.size(negative))

    #if debug==1: print( np.sum(negative>120)  )

    #sleep(1

    #if debug==1: print(frame)

    #if debug==1: print(prev_frame)

    #raise





    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    faces = faceCascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(30, 30)

    )

    #print(faces)

    #print(faces.count())



    # Draw a rectangle around the faces
    face_count=0
    name="Not Recognized Person"
    for (x, y, w, h) in faces:
        face_count+=1
        facetime1=time()
        #print("face_count",face_count,time())

        #resized = cv2.resize(frame[y:y+h,x:x+w], (160,160), interpolation = cv2.INTER_AREA)

        #if debug==1: grayplt(resized/255)

        #numm+=1

        #saved=cv2.resize(frame[y:y+h,x:x+w], (160,160), interpolation = cv2.INTER_AREA)

        gamma=1

        #imgg=cv2.imread("./img/ad2.jpg")

        #imgg=adjust_gamma(imgg, gamma=0.8)

        

        #Full image: frame

        tim5=time()
        success,imag=image_process(cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2RGB),gamma)  
        tim6=time()
        tim=tim6-tim5
        print("preprocessing_time:",tim)

        #success,imag=image_process(cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB),gamma)   

        if success==False: continue

        #imag=frame[y:y+h,x:x+w]

        encode1=time()
        face_locations = face_recognition.face_locations(imag, number_of_times_to_upsample=2, model='hog') # For GPU, use model='cnn'

        face_encodings = face_recognition.face_encodings(imag, face_locations, num_jitters=2)
        encode_time=time()-encode1
        print("encoding time:", encode_time)

        #if debug==1: print(face_locations)                

        #if debug==1: print(face_encodings)        

        if face_encodings==[]: continue             

        

        cv2.imwrite('saved.jpg',frame[y:y+h,x:x+w])

        if debug==1: grayplt(imag/255)

        if debug==1: print("image updated as saved.jpg")

        resized=np.expand_dims(imag,axis=0)



        img2_representation=np.expand_dims(face_encodings[0],axis=0)
        
        if debug==1:
            result2=model3.predict(img2_representation)    
            result3=model4.predict(img2_representation)    
            result4=model5.predict(img2_representation)    
            result5=model6.predict(img2_representation)

            result6=model7.predict(img2_representation)    
            result7=model8.predict(img2_representation)    
            result8=model9.predict(img2_representation)    
            result9=model10.predict(img2_representation)
    
            result10=model11.predict(img2_representation)    
            result11=model12.predict(img2_representation)
            result12=model13.predict(img2_representation)    
            result13=model14.predict(img2_representation)

            result18=model19.predict(img2_representation)
            result19=model20.predict(img2_representation)
            result20=model21.predict(img2_representation)
            result21=model22.predict(img2_representation)

        tim1=time()
        result14=model15.predict(img2_representation)
        result15=model16.predict(img2_representation)
        result16=model17.predict(img2_representation)
        result17=model18.predict(img2_representation)
        tim2=time()
        tim=tim2-tim1
        print("voting time:",tim)
        #print(img2_representation)



        

        if debug==1: print(img2_representation.shape)
        if debug==1: print("mlp")
        if debug==1: print("francis",result2)
        if debug==1: print("Yu Ka",result3)
        if debug==1: print("boonping",result4)
        if debug==1: print("not recognized",result5)

        if debug==1: print("lr")
        if debug==1: print("francis",result6)
        if debug==1: print("Yu Ka",result7)
        if debug==1: print("boonping",result8)
        if debug==1: print("not recognized",result9)



        if debug==1: print("knn")

        if debug==1: print("francis",result10)

        if debug==1: print("Yu Ka",result11)

        if debug==1: print("boonping",result12)

        if debug==1: print("not recognized",result13)



        if debug==1: print("svm")

        if debug==1: print("francis",result18)

        if debug==1: print("Yu Ka",result19)

        if debug==1: print("boonping",result20)

        if debug==1: print("not recognized",result21)



        if debug==1: print("voting")

        if debug==1: print("francis",result14)

        if debug==1: print("Yu Ka",result15)

        if debug==1: print("boonping",result16)

        if debug==1: print("not recognized",result17)

        

        '''

        cosine = findCosineDistance(img1_representation, img2_representation)

        euclidean = findEuclideanDistance(img1_representation, img2_representation)

        

        if cosine <= 0.02:

           if debug==1: print("this is boonping")

        else:

           if debug==1: print("this is not boonping")

        '''

        ########
        if debug==1:
            prediction=model2.predict(img2_representation)
    
    
            fa=-1
            val=0
            sel=0
    
            for fac in range(3):
                fa+=1
                if np.argmax(prediction[fac][0])==1:
                    if fa==0 and prediction[fac][0][1]>val: 
                        sel=1
                        val=prediction[fac][0][1]
    
                    if fa==1 and prediction[fac][0][1]>val: 
                        sel=2
                        val=prediction[fac][0][1]                    
    
                    if fa==2 and prediction[fac][0][1]>val: 
                        sel=3
                        val=prediction[fac][0][1]
    
                    if fa==3 and prediction[fac][0][1]>val: 
                        sel=4
                        val=prediction[fac][0][1]
    
            if val<0.55 and sel!=4: sel=0
    
            
    
            if debug==1: print("\nResult with deep neural network")
            print("result14: ",result14)
    
            if sel==0: 
    
                if debug==1: print("not recognized")
    
            elif sel==1: 
    
                if result14==1 and result15==0  and result16==0:
    
                    if min(model11.kneighbors(img2_representation,return_distance=True)[0][0])>3.0:
    
                        if debug==1: print("Not sure it is Francis")
    
                    else:
    
                        if debug==1: print("Francis")
    
                        
    
                else:
    
                    if debug==1: print("not recognized")
    
            elif sel==2: 
    
                if result15==1 and result14==0  and result16==0:            
    
                    
    
                    if min(model11.kneighbors(img2_representation,return_distance=True)[0][0])>3.0:                    
    
                        if debug==1: print("Not sure it is Yu Ka")
    
                    else:
    
                        if debug==1: print("Yu Ka")
    
                    
    
                else:
    
                    if debug==1: print("not recognized")
    
                    
    
            elif sel==3: 
    
                if result16==1 and result15==0  and result14==0:
    
                    
    
                    if min(model11.kneighbors(img2_representation,return_distance=True)[0][0])>3.0:
    
                        if debug==1: print("Not sure it is Boon Ping")
    
                    else:
    
                        if debug==1: print("Boon Ping")
    
                        name="Boon Ping"
    
                        
    
                    
    
                    
    
                else:
    
                    if debug==1: print("not recognized")
    
                    
    
            elif sel==4: 
    
                if debug==1: print("not recognized")
        ######

        #cv2.imwrite('saved_%i_%i.jpg'%(sel,numm),saved)

        if debug==1: print("\nResult with voting network only")    
        anatime1=time()
        name="Not Recognized Person"

        if result14[0]==1 and result15[0]==0  and result16[0]==0: # and model11.predict_proba(img2_representation)[1]>0.666:
            '''
            if min(model11.kneighbors(img2_representation,return_distance=True)[0][0])>3.0:
                if debug==1: print("Not sure it is Francis")
            else:
                if debug==1: print("Francis")
                name="Francis"
                '''
            name="Francis"

        elif result15[0]==1 and result14[0]==0  and result16[0]==0: # and model12.predict_proba(img2_representation)[1]>0.666:            
            '''
            if min(model11.kneighbors(img2_representation,return_distance=True)[0][0])>3.0:                    
                if debug==1: print("Not sure it is Yu Ka")
            else:
                if debug==1: print("Yu Ka")
                name="Yu Ka"
            '''
            name="Yu Ka"

        elif result16[0]==1 and result15[0]==0  and result14[0]==0: # and model13.predict_proba(img2_representation)[1]>0.666:
            '''
            if min(model11.kneighbors(img2_representation,return_distance=True)[0][0])>3.0:
                if debug==1: print("Not sure it is Boon Ping")
            else:
                if debug==1: print("Boon Ping")
                name="Boon Ping"
            '''
            name="Boon Ping"
            

        else:
            if debug==1: print("Not recognized")
        analyze_time=time()-anatime1
        print("analyze_time",analyze_time)
            
        #print(result14)
        #(top, right, bottom, left)=face_locations[0]

        

        #Full image: processing full frame

        #unmask to get full frame: 

        #cv2.imwrite('saved_frame.jpg',frame)

        

        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        tim3=time()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x + 6, y+h - 6), font, 1.0, (0, 0, 255), 1)
        changed=1
        tim4=time()
        tim=tim4-tim3
        print("rectangle time:",tim)
        facetime=time()-facetime1
        print("per face time:",facetime)
        if debug==1: print(min(model11.kneighbors(img2_representation,return_distance=True)[0][0]))

            

        


        

        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #sleep(0.01)


    # Display the resulting frame

    if changed==1:
        tim7=time()
        cv2.imshow('Video', frame)
        tim8=time()
        tim=tim8-tim7
        print("video time: ",tim)
        time2=time()
        frame_time=time2-time1
        print("frame time is: ", frame_time)
        sleep(0.5)
        changed=0
        static_count=0

    else:        
        static_count+=1
        time2=time()
        frame_time=time2-time1
        print("frame time is: ", frame_time)
        if static_count>1:
            cv2.imshow('Video', frame)
            changed=0
            static_count=0

    #print(faces)

    grayplt(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(1) & 0xFF == ord('q'):

        '''

        image = load_img("frame2.jpg")

        image = img_to_array(image)

        image = np.expand_dims(image, axis=0)



        aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

        if debug==1: print("[INFO] generating images...")

        imageGen = aug.flow(image, batch_size=1, save_to_dir=".",save_prefix="image5", save_format="jpg")

        i=0

        for image in imageGen:

            if debug==1: print(image)

            i+=1

            if i==100: break

        '''

        break



    # Display the resulting frame

    #cv2.imshow('Video', frame)



# When everything is done, release the capture

video_capture.release()

cv2.destroyAllWindows()
