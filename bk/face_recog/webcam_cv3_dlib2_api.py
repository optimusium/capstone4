import logging as log
import os
import pickle
from time import time

import cv2
import numpy as np
from keras.models import load_model

from mtcnn import MTCNN
import face_recognition
# import face_recognition
from bk.face_recog.utils import image_process2

debug=0

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


