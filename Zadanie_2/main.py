
import cv2 as cv
import numpy as np
import sys
# from ximea import xiapi

def main():
    # print(cv.getBuildInformation())

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

    # define size of grid
    grid = np.zeros((5 * 7, 3), np.float32)

    # create 2,7,6 array, transpose to 5,7,2 and reshape to 5*7,2
    grid[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)

    # create arrays to store data
    objpoints = []  # 3d point in real world
    imgpoints = []  # 2d points on the pictures

    for x in range(1, 15):

        # path to image folder
        path = "images2/img" + str(x) + ".jpg"
        image = cv.imread(path)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (7, 5), None)

        if ret == True:
            objpoints.append(grid)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(image, (7, 5), corners2, ret)
            cv.imshow('img', image)
            cv.imwrite("images2/img" + str(x) + "_lineDetect.png", image)
            cv.waitKey(500)

    cv.destroyAllWindows()

    # # calibration
    # # in order: return value, camera matrix, distortion coefficients, rotation and translation vectors
    #
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(mtx)

    # image = cv.imread("images2/img6.jpg")
    # h, w = image.shape[:2]
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    #
    # # undistort
    # dst = cv.undistort(image, mtx, dist, None, newcameramtx)
    # # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]
    # cv.imwrite("images2/caibration_result.png", dst)

        # cv.imshow("temp", image)
        # cv.waitKey()


    # Detection of the circle
    path2 = "images2/circle3.jpg"

    # Loads an image
    img = cv.imread(cv.samples.findFile(path2), cv.IMREAD_COLOR)

    # Convert it to gray
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Reduce the noise to avoid false circle detection
    img_gray = cv.medianBlur(img_gray, 5)

    # Applying Hough Circle Transform
    rows = img_gray.shape[0]
    circle_coordinates = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=70, maxRadius=150)

    # Drawing detected circles
    if circle_coordinates is not None:
        circle_coordinates = np.uint16(np.around(circle_coordinates))
        for i in circle_coordinates[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(img, center, radius, (255, 0, 255), 3)

    cv.imshow("detected circles", img)
    cv.waitKey(0)

if __name__ == '__main__':
    main()

