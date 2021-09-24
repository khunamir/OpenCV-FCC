import cv2 as cv

img = cv.imread('../Photos/park.jpg')
cv.imshow('Park', img)

# Converting to grayscale
grayscaled = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscaled', grayscaled)

# Blur
blurred = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blurred', blurred)

# Edge Cascade
canny = cv.Canny(blurred, 100, 125)
cv.imshow('Canny Edges', canny)

# Dilating
dilated = cv.dilate(canny, (5,5), iterations=3)
cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (5,5), iterations=3)
cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (400,400), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Cropping
cropped = img[100:200,100:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)