# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:33:50 2020

@author: Cao Liang
"""

import cv2
import logging as log
from time import sleep
from time import time

from face_recognition_api.webcam_cv3_dlib2_api import FaceRecognizer

log.basicConfig(level=log.DEBUG,
                    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                    handlers=[
                        log.FileHandler("logs/face_recog_client.log"),
                        log.StreamHandler()
                    ])

def run_model():

    try: 
        
        #model2=createModel()
        face_recognizer = FaceRecognizer()
        
        video_capture = cv2.VideoCapture(0)
        
        img_index = 0
        changed=1
        static_count=0
        frame_time=0
    
        while True:
            if not video_capture.isOpened():
                log.error('Unable to load camera.')
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
        
            faces_list = face_recognizer.perform_face_recognition(frame)
            face_index = 0
            
            for faces in faces_list:
                face_show_time = time()
                
                face_index += 1
                face_name = faces[0]
                x, y, w, h = faces[1]
                face_img = faces[2]
                
                log.debug(f"Located face: {face_index}")
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, face_name, (x + 6, y+h - 6), font, 1.0, (0, 0, 255), 1)
                
                changed=1
                
                time_spent = time() - face_show_time
                log.debug("face show time: {}".format(time_spent))
                
                img_index += 1
                # Save face image for test
                cv2.imwrite('{}_{}.png'.format(face_name, img_index), face_img)
                
                #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                sleep(0.01)
        
            # Display the resulting frame
        
            if changed==1:
                tim7=time()
                cv2.imshow('Video', frame)
                tim8=time()
                tim=tim8-tim7
                log.debug("video time:  {}".format(tim))
                time2=time()
                frame_time=time2-time1
                log.debug("frame time is: {}".format(frame_time))
                #sleep(0.5)
                changed=0
                static_count=0
        
            else:        
                static_count+=1
                time2=time()
                frame_time=time2-time1
                log.debug("frame time is: {}".format(frame_time))
                if static_count>1:
                    cv2.imshow('Video', frame)
                    changed=0
                    static_count=0
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
            # Display the resulting frame
            #cv2.imshow('Video', frame)
    finally:
        # When everything is done, release the capture
        try: 
            if video_capture is not None:
                video_capture.release()     
        except Exception:
            pass
    
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__=="__main__":
    run_model()