import cv2 as cv

# Reading Images

img = cv.imread('Photos/cat.jpg')

cv.imshow('Cat', img)

cv.waitKey(0)

# Reading Videos

capture = cv.VideoCapture('Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    cv.imshow('Dog', frame)

    if cv.waitKey(30) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()