# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:33:50 2020

@author: Cao Liang
"""

import cv2
import logging as log
from time import sleep
from time import time
from datetime import datetime
import argparse

from face_recognition_api.webcam_cv3_dlib2_api import FaceRecognizer
from backend_service.AlertClient import send_email_with_images as send_email
from backend_service.AlertClient import send_sms
from backend_service.AlertClient import query_intruder_status


def monitor_access():
    log.basicConfig(level=log.DEBUG,
                format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                handlers=[
                    log.FileHandler("logs/access_monitor.log"),
                    log.StreamHandler()
                ])

    try: 
        parser = argparse.ArgumentParser(description='This is Security access monitor application. You can process both videos and images.')
        parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='0')
        parser.add_argument('--phone', type=str, help='sms_target_phone', default='0')
        args = parser.parse_args()
        
        face_recognizer = FaceRecognizer()
        sms_phone = args.phone
        
        if args.input == "0":
            video_capture = cv2.VideoCapture(int(args.input))
        else:
            video_capture = cv2.VideoCapture(args.input)
        
        img_index = 0
        changed=1
        static_count=0
        frame_time=0
        sms_sent_time = 0
        email_sent_time = 0
        
    
        while True:
            if not video_capture.isOpened():
                log.error('Unable to load camera.')
                sleep(5)
                pass
        
            # Capture frame-by-frame
            ret, frame = video_capture.read()
        
            intrusion_found = query_intruder_status()
    
            if not intrusion_found:
                log.debug("No finding")
                cv2.imshow('Video', frame)
                
                sleep(0.25)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
                continue
                
            log.warning("Intrusion found!")
            
            #cv2.imwrite("frame.jpg",frame)
            #sleep(0.25)
        
            if changed==1:
                cv2.imshow('Video', frame)
                changed=0
            
            time1=time()
        
            faces_list = face_recognizer.perform_face_recognition(frame)
            face_index = 0
            
            unkown_face_images = []
            
            for faces in faces_list:
                face_index += 1
                face_name = faces[0]
                x, y, w, h = faces[1]
                face_img = faces[2]
                
                log.debug(f"Located face: {face_index}, name: {face_name}")
                img_index += 1
                if face_name == "unknown":
                    # Save unknown face image for audit
                    #cv2.imwrite('logs/unkown_faces/{}_{}.png'.format(face_name, img_index), face_img)
                    unkown_face_images.append(face_img)

            num_unkown_faces = len(unkown_face_images)
            if num_unkown_faces > 0:
                log.info(f"Unkown face found: {num_unkown_faces}")
                current_time = datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
                # Sent SMS alert
                sms_message = "Detection Alert - Unauthorized {} person: {}".format(
                        num_unkown_faces, current_time) 
                
                # To avoid SMS sending too much, only send sms every 5 minutes
                last_sms_sent_time = sms_sent_time
                sms_wait_time = time() - last_sms_sent_time
                if last_sms_sent_time == 0 or sms_wait_time >= 300:
                    if sms_phone == "0":
                        send_sms_result = send_sms(sms_message)    
                    else:
                        send_sms_result = send_sms(sms_message, 
                                                   target_phone=sms_phone)
                    
                    log.info("Sending SMS alert(result: {}): {}".format(
                            send_sms_result, sms_message))
                    sms_sent_time = time()
                else:
                    log.info(f"Not sending SMS due to waiting time < 5 min")
                
                # To avoid email sending too much, only send sms every 1 minutes
                last_email_sent_time = email_sent_time
                email_wait_time = time() - last_email_sent_time
                if last_email_sent_time == 0 or email_wait_time >= 60:
                    # Sent Email alert
                    email_content = "Unauthorized {} person: {}".format(
                            num_unkown_faces, current_time)
                    email_subject = "Unauthorized Person Detected"
                    send_email(email_content, email_subject, "Detection Alert",
                               image_name_prefix="unknown_", 
                               image_list=unkown_face_images)
                    log.info("Sending SMS (result: {}) to alert: {}".format(
                                send_sms_result, sms_message))
                    
                    send_email_result = send_sms(sms_message)
                    log.info("Sending Email alert (result: {}): {}".format(
                            send_email_result, email_content))
                    email_sent_time = time()
                else:
                    log.info(f"Not sending email due to waiting time < 1 min")
                
        
            for faces in faces_list:
                face_show_time = time()
                
                face_index += 1
                face_name = faces[0]
                x, y, w, h = faces[1]
                face_img = faces[2]
                
                log.debug(f"Located face to show: {face_index}, name: {face_name}")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, face_name, (x + 6, y+h - 6), font, 1.0, (0, 0, 255), 1)
                
                changed=1
                
                time_spent = time() - face_show_time
                log.debug("face show time: {}".format(time_spent))
        
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
    monitor_access()