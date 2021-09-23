import cv2 as cv
import numpy as np

# Draw a blank page
blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow('Blank', blank)

# 1. Draw a image a certain color
blank[150:400, 150:400] = 0,0,255
cv.imshow('Green', blank)

# 2. Draw a rectangle
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), -1) 
cv.imshow('Rectangle', blank)

# 3. Draw a circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 50, (0,0,255), -1)
cv.imshow('Circle', blank)

# 4. Draw a line
cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (10,10,10), 2)
cv.imshow('Line', blank)

# 5. Write a text
cv.putText(blank, 'Hello World ', (10, blank.shape[0]//2), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), 2)
cv.putText(blank, 'from', (10, blank.shape[0]//2 + 40), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), 2)
cv.putText(blank, 'Amir Fahmy!!!', (10, blank.shape[0]//2 + 80), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), 2)
cv.imshow('Hello World', blank)

cv.waitKey(0)