#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import face_recognition

CUR_PATH = '/Users/yanjingang/project/face_recognition'

print(time.time())
image = face_recognition.load_image_file(CUR_PATH+"/data/infer/1.png")
print(time.time())
emb = face_recognition.face_encodings(image)[0]
print(time.time())
face_locations = face_recognition.face_locations(image)
print(face_locations)
print(time.time())

face_landmarks_list = face_recognition.face_landmarks(image)
print(face_landmarks_list)
print(time.time())


img_yjy = face_recognition.load_image_file(CUR_PATH+"/data/face/yjy.png")
emb_yjy = face_recognition.face_encodings(img_yjy)[0]
print(emb_yjy)
print(time.time())

img = face_recognition.load_image_file(CUR_PATH+"/data/face/yan.png")
emb_yan = face_recognition.face_encodings(img)[0]
print(emb_yan)
print(time.time())

img = face_recognition.load_image_file(CUR_PATH+"/data/face/zhu.png")
emb_zhu = face_recognition.face_encodings(img)[0]
print(emb_zhu)
print(time.time())


known_face_encodings = [emb_yjy,emb_yan,emb_zhu]

distances = face_recognition.face_distance(known_face_encodings, emb)
print(distances)
print(list(distances<=0.4))
print(time.time())


results = face_recognition.compare_faces(known_face_encodings, emb, tolerance=0.4)
print(results)
print(time.time())


