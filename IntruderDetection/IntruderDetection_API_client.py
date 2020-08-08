# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:14:51 2020

@author: Cao Liang
"""

import cv2 as cv
import logging as log
from time import sleep
import argparse
from datetime import datetime

from IntruderDetection.IntruderDetection_API import IntrusionDetector

log.basicConfig(level=log.DEBUG,
                format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                handlers=[
                    log.FileHandler("./logs/intrusion_detect_client.log"),
                    log.StreamHandler()
                ])


def detect_intrusion():
    
    try:
        parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                      OpenCV. You can process both videos and images.')
        parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='0')
        parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
        args = parser.parse_args()
        
        
        detector = IntrusionDetector(algo=args.algo)
        
        if args.input == "0":
            capture = cv.VideoCapture(int(args.input))
        else:
            capture = cv.VideoCapture(args.input)
        
        if not capture.isOpened:
            log.error('Unable to open: ' + args.input)
            exit(0)
               
        while True:
            ret, frame = capture.read()
            if frame is None:
                break
            
            intrusion_found, fgMask = detector.perform_intrusion_detection(frame)
    
            if intrusion_found:
                log.warning("Intrusion found!" + str(datetime.now()))
            else:
                log.debug("No finding")
                
            #cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            #cv.rectangle(frame, None, None, (255, 255, 255), -1)
            
            #cv.imshow('Frame', frame)
            cv.imshow('FG Mask', fgMask)
            
            # fgmask = fgbg.apply(frame)
            # erosion and dilation
            # fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        
            #print("fgMask:", fgMask)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Wait for 250ms
            sleep(0.25)
    finally:
        # When everything is done, release the capture
        try: 
            if capture is not None:
                capture.release()     
        except Exception:
            pass
    
        try:
            cv.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    detect_intrusion()