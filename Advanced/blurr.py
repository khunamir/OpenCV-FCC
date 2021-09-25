import cv2 as cv

img = cv.imread('../Photos/cats.jpg')
cv.imshow('Cats', img)

# Averaging Blur
average = cv.blur(img, (7,7))
cv.imshow('Average', average)

# Gaussian Blur
gauss = cv.GaussianBlur(img, (7,7), 0)
cv.imshow('Gaussian', gauss)

# Median Blur
median = cv.medianBlur(img, 7)
cv.imshow('Median', median)

# Bilateral Blue
bilateral = cv.bilateralFilter(img, 10, 15, 15)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)