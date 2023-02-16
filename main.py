

import cv2 as cv
import keyboard

cam = cv.VideoCapture(0)

print("Press space to begin")
keyboard.wait(" ")
print("Starting")

for x in range(4):
    res, image = cam.read()
    if res:
        temp = str(x)
        path = "pic"+temp+".png"
        cv.imwrite(path, image)
    else:
        print("Error reading the image")
