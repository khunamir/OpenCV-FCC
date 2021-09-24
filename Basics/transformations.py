import cv2 as cv
import numpy as np

img = cv.imread('../Photos/park.jpg')

# Translate

def translate(img, x, y):
    # -x --> Left
    # -y --> Up
    # x --> Right
    # y --> Down
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

translated = translate(img, 40, 40)
cv.imshow('Translated', translated)

# Rotation

def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, 45)
cv.imshow('Rotated', rotated)

# Resizing
resized = cv.resize(img, (250, 250), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Flipping
flipped = cv.flip(img, 0)
cv.imshow('Flipped', flipped)

# Cropping
cropped = img[200:300,300:450]
cv.imshow('Cropped', cropped)

cv.waitKey(0)