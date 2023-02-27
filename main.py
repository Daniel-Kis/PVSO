
import cv2 as cv
import numpy as np
import os

def main():
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
    cv.imwrite("mozaik_complete_image.png", complete_image)

    # kernel mask definition
    kernel = np.array([[0, -1, 0], [-5, 6, 0], [1, 0, 1]], np.float32)

    # apply the mask only for the first image
    mask = cv.filter2D(complete_image[0:480, 0:640], -1, kernel)

    # replace the first image of original photo with the new one
    complete_image[0:480, 0:640] = mask
    cv.imshow("Output", complete_image)
    cv.waitKey(0)

    # save the mozaik with masked image
    cv.imwrite("mozaik_mask_complete_image.png", complete_image)

    # rotation of image
    # reading of image and geting its dimensions
    img = cv.imread("mozaik_mask_complete_image.png",1)
    img2 = (img[0:480, 640:1280])
    rows,cols,ht = img2.shape

    # Creating new matrix of same tipe as image img using image helpI.png for help
    mask2 = (img[0:640, 0:480])
    cv.imwrite("helpI.png", mask2)
    mask3 = cv.imread("helpI.png", 1)

    # rotating image and then resize it, so it would fit back into original image through mask3
    for i in range(rows-1):
        mask3[:, i] = img2[i, :]

    mask3 = cv.resize(mask3, (640, 480))

    # this is original method
    # # creation of rotation metrix
    # matrix = cv.getRotationMatrix2D((rows/2,cols/2),90,1)
    #
    # # rotating image and then resize it so it would fit back into original through mask2
    # mask2 = cv.warpAffine(img2,matrix,(rows,cols))
    # mask2 = cv.resize(mask2, (640,480))

    # replace the second image of original photo with the new one
    img[0:480, 640:1280] = mask3
    cv.imshow("output",img)

    # saving new image
    cv.imwrite("mozaik_rotated_complete_image.png",img)

    cv.waitKey(0)

    # Changing color chanel to R
    # reading of image
    img = cv.imread("mozaik_rotated_complete_image.png")

    # creating mask for R chanel
    mask3 = (img[480:960, 0:640])
    mask3[:, :, 0] = 0
    mask3[:, :, 1] = 0

    # replace the second image of original photo with the new one
    #   mask3 = cv.resize(mask3, (480, 640))
    img[480:960, 0:640] = mask3
    cv.imshow("output", img)

    # saving new image
    cv.imwrite("mozaik_red_channel_complete_image.png", img)

    cv.waitKey(0)

    # terminal output
    # reading of image and geting its dimensions
    img = cv.imread("mozaik_red_channel_complete_image.png", 1)
    rows, cols, ht = img.shape
    dt = (".png")
    size = os.path.getsize("mozaik_red_channel_complete_image.png")

    # printing information
    print("data type: "+str(dt)+"\ndimensions X: "+str(cols)+" Y: "+str(rows)+"\nsize: "+str(size)+" B")

    cv.waitKey(0)

    return 0


if __name__ == "__main__":
    main()
