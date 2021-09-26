import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('../Photos/cats.jpg')

blank = np.zeros(img.shape[:2], dtype='uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

circle = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)

masked = cv.bitwise_and(gray, gray, mask=circle)
cv.imshow('Mask', masked)

gray_hist = cv.calcHist([gray], [0], masked, [256], [0,256])

# Grayscale histogram
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('Pixel Density')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

# Color histogram
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('Pixel Density')
color = ('b','g','r')
for i,col in enumerate(color):
    hist = cv.calcHist([img], [i], circle, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.show()

cv.waitKey(0)