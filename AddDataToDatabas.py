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


while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations((imgS))
    encodeCurrFrame = face_recognition.face_encodings(imgS,faceCurrFrame)

    imgBackground[162:162+480,55:55+640] = img
    imgBackground[44:44+633,808:808+414] = imgModeList[modeType]

    if faceCurrFrame:
        for encodeFace, faceLoc in zip(encodeCurrFrame,faceCurrFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            #print("matches", matches)
            #print("distance", faceDis)

            matchIndex = np.argmin(faceDis)
            #print("matchIndex", matchIndex)

            if matches[matchIndex]:
                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
                bbox = 55+x1, 162+y1, x2-x1, y2-y1
                imgBackground = cvzone.cornerRect(imgBackground,bbox,rt=0)
                id = studentIds[matchIndex]
                #print(id)

                if counter==0:
                    cvzone.putTextRect(imgBackground, "Loading...",(275,400))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    counter =1
                    modeType = 1
            if counter!=0:

