import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('../Photos/park.jpg')

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()

# HSV to BGR
hsv_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
cv.imshow('BGR', hsv_bgr)

cv.waitKey(0)