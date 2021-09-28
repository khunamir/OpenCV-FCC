import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recog_model.yml')

img = cv.imread("val/ben_afflek/5.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Face', gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label: {people[label]} with a confidence of {confidence}')

    cv.putText(img, people[label], (15,15), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 1)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

cv.imshow('Detected Face', img)

cv.waitKey(0)