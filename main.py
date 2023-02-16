

import cv2 as cv

cam = cv.VideoCapture(0)

x = 1

while x <= 4:
    res, image = cam.read()
    if res:
        cv.imshow("camera", image)
        if cv.waitKey() == ord(' '):
            temp = str(x)
            path = "pic" + temp + ".png"
            cv.imwrite(path, image)
            x = x + 1
            if x > 4:
                break
    else:
        print("Error reading from camera")