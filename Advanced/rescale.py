import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    # Only for live videos eg:- webcam
    capture.set(3, width)
    capture.set(4, height)

capture = cv.VideoCapture('Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame, scale=.5)

    cv.imshow('Dog', frame)
    cv.imshow('Resized Dog', frame_resized)

    if cv.waitKey(30) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()