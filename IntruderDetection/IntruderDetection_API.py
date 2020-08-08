# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:14:51 2020

@author: Cao Liang
"""

import cv2 as cv
import logging as log


log.basicConfig(level=log.DEBUG,
                format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                handlers=[
                    log.FileHandler("logs/intrusion_detection_client.log"),
                    log.StreamHandler()
                ])

class IntrusionDetector(object):
    
    def __init__(self, algo="KNN", threshold=0.93):
        if algo == 'MOG2':
            self.backSub = cv.createBackgroundSubtractorMOG2()
        else:
            self.backSub = cv.createBackgroundSubtractorKNN()

        self.threshold = threshold
        
        
    def perform_intrusion_detection(self, frame):
        """
        Parameters:
             frame: Open CV captured frame image or video file frame image.
        
        Return:
             List of detection result and mask image; Detection result is true 
             if intrusion is detected, and False if not detected.
        """
        detected = False
        
        fgMask = self.backSub.apply(frame)
        valueDict = {}
        for i in range(len(fgMask)):
            for j in range(len(fgMask[i])):
                # print("type:", type(fgMask[i][j]))
                x = valueDict.get(fgMask[i][j])
                if x == None:
                    valueDict[fgMask[i][j]] = 1
                else:
                    valueDict[fgMask[i][j]] = valueDict[fgMask[i][j]] + 1
        
        #print("valueDict:", valueDict)
    
        if valueDict.get(0) is not None:
            if valueDict[0] / fgMask.size < self.threshold:
                log.info("Intrusion detected !")
                detected = True

        return [detected, fgMask]

