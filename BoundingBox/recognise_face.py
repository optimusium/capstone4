# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:23:23 2020

@author: Jacky
"""

#code forked and tweaked from https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
#to extend, just add more people into the known_people folder

import face_recognition
import cv2
import numpy as np
import os
import glob
import argparse

parser = argparse.ArgumentParser(description='This module recognizes faces with a named bounding box around the face.')
parser.add_argument('--camera', type=int, help='Which camera to use.', 
                    default=0)
parser.add_argument('--width', type=int, help='Which frame width to use.', 
                    default=1920)
parser.add_argument('--hight', type=int, help='Which frame hight to use.', 
                    default=1080)
parser.add_argument('--xResize', type=int, help='How much to resize image(x).', 
                    default=2)
parser.add_argument('--yResize', type=int, help='How much to resize image(y).', 
                    default=2)
parser.add_argument('--upsample', type=int, help='Number of times to upsample the image looking for faces. Higher numbers find smaller faces.', 
                    default=2)
parser.add_argument('--model', type=str, help='Which face detection model to use.', 
                    default='hog')
parser.add_argument('--jitter', type=int, help='Number of times to re-sample the face when calculating encoding. Higher is more accurate, but slower.', 
                    default=2)
parser.add_argument('--tolerance', type=float, help='How much distance between faces to consider it a match. Lower is more strict.', 
                    default=0.6)

args = parser.parse_args()


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(args.camera)

# Increase the resolution of the video
def setResolution(width, hight):
    """Set resolution of the video input"""
    video_capture.set(3, width)
    video_capture.set(4, hight)

setResolution(args.width, args.hight)

#make array of sample pictures with encodings
known_face_encodings = []
known_face_names = []
dirname = os.path.dirname(__file__)
path = os.path.join(dirname, 'known_people/')

#make an array of all the saved jpg files' paths
list_of_files = [f for f in glob.glob(path+'*.jpg')]
#find number of known faces
number_files = len(list_of_files)

names = list_of_files.copy()

for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    known_face_encodings.append(globals()['image_encoding_{}'.format(i)])

    # Create array of known names
    names[i] = names[i].replace("known_people\\", "").replace(".jpg", "")  
    known_face_names.append(names[i])

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Resize frame of video to 2 times the size for larger display
    frame = cv2.resize(frame, (0,0), fx=args.xResize, fy=args.yResize) 

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=args.upsample, model=args.model) # For GPU, can try model='cnn'
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=args.jitter)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=args.tolerance) # Lower is more strict, default = 0.6
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations by 2 times since the frame we detected in was scaled to 1/4 size
        top *= 4*args.xResize
        right *= 4*args.xResize
        bottom *= 4*args.xResize
        left *= 4*args.xResize

        # Draw a box around the face
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 5.0, (0, 0, 0), 5)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()