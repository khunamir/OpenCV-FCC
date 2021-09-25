import cv2 as cv
import numpy as np

img = cv.imread('../Photos/cats 2.jpg')
cv.imshow('Cats 2', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank', blank)

circle = cv.circle(blank.copy(), (img.shape[1]//2-100, img.shape[0]//2-75), 150, 255, -1)

rectangle = cv.rectangle(blank.copy(), (100,25), (325,250), 255, -1)

weird_shape = cv.bitwise_and(rectangle, circle)

masked = cv.bitwise_and(img, img, mask=weird_shape)
cv.imshow('Masked Image', masked)

cv.waitKey(0)