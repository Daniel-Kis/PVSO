

import cv2 as cv

# initialize the webcam
cam = cv.VideoCapture(0)

x = 1

while x <= 4:

    # read the result of the operation and get the image from camera
    res, image = cam.read()

    # success
    if res:

        # show image on screen in separate window
        cv.imshow("camera", image)

        # wait until space is pressed
        if cv.waitKey() == ord(' '):

            # this part is for periodically changing the name of images
            temp = str(x)
            path = "pic" + temp + ".png"

            # save image
            cv.imwrite(path, image)

            # increase loop control variable
            x = x + 1

            # if 4 images were taken, break from loop
            if x > 4:
                break
    # in case of failed read from camera
    else:
        print("Error reading from camera")

# load the images that were taken in the previous part
image1 = cv.imread("pic1.png")
image2 = cv.imread("pic2.png")
image3 = cv.imread("pic3.png")
image4 = cv.imread("pic4.png")

# first, vertically concentrate them
vertical_con1 = cv.vconcat([image1, image2])
vertical_con2 = cv.vconcat([image3, image4])

# then horizontally
complete_image = cv.hconcat([vertical_con1, vertical_con2])

# the final mozaik
cv.imshow("Complete image", complete_image)
cv.waitKey()