#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import copy
import face_recognition
import cv2

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))


known_face_encodings, known_face_names = [], []
facedb_path = CUR_PATH+'/data/faceid/'
for img_file in os.listdir(facedb_path):
    if img_file == '.DS_Store':
        continue
    print(facedb_path +  img_file)
    image = face_recognition.load_image_file(facedb_path +  img_file)
    emb = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(emb)
    known_face_names.append(img_file.split('.')[0])
    #known_face_names.append(img_file)
print(known_face_names)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

video_capture = cv2.VideoCapture(0)
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    #rgb_small_frame = frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            '''
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            '''
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            #print(distances)
            if distances is not None:
                sort_distances = copy.deepcopy(distances)
                #print(sort_distances)
                sort_distances.sort()
                #print(sort_distances)
                idx = list(distances).index(sort_distances[0])
                #print(idx)
                name = known_face_names[idx]
                #print(name)
            
            face_names.append(name)

    process_this_frame = not process_this_frame
    #print(face_locations)
    #print(face_names)


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 255, 0), 1)
        cv2.putText(frame, name, (left + 6, top + 12), font, 0.5, (0, 255, 0), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
