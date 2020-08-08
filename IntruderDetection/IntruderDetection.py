# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:03:28 2020

@author: Jacky
"""

from __future__ import print_function
import cv2 as cv
import argparse
from datetime import datetime
from backend_service import AlertClient

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture(int(args.input))
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)


lastTriggerTime = datetime.now()
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
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
        if valueDict[0] / fgMask.size < 0.98:
            # print("trigger========================", datetime.now())
            currentTime = datetime.now()
            timeDiff = currentTime - lastTriggerTime
            if timeDiff.seconds > 1:
                lastTriggerTime = datetime.now()
                AlertClient.inform_intruder_status(True)
            #else:
                # print("trigger ignored since it is within 1s compare to last one", timeDiff.microseconds)

    #cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.rectangle(frame, None, None, (255, 255, 255), -1)

    #cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    # fgmask = fgbg.apply(frame)
    # erosion and dilation
    # fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

    # cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    #print("fgMask:", fgMask)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
