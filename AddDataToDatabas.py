import os
import pickle
from typing import List
import numpy as np
import cvzone
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://faceattendance-28ec5-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendance-28ec5.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
imgBackground =  cv2.imread('Resources/background.png')

folderModePath = 'Resources/Modes'
modePathList: list[str] = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))
#print(len(imgModeList))

#loading encoding of images from the pickle file
print("Loading Encoded file ...")
file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
#print(studentIds)
#print("Loaded encode file")

modeType = 0
counter = 0
id = 0
imgStudent = []

