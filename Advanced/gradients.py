import cv2 as cv
import numpy as np

img = cv.imread('../Photos/park.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Laplacian
laplacian = cv.Laplacian(gray, cv.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))
cv.imshow('Laplacian', laplacian)

# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)

# Canny
canny = cv.Canny(gray, 125, 175)
cv.imshow('Canny', canny)

cv.waitKey(0)