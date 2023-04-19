import cv2 as cv
import numpy as np
import ctypes as ct


def ptr2d_to_mat(ptr, rows, cols):
    return tuple(tuple(ptr[i][j] for j in range(cols)) for i in range(rows))


def laplaceEdgeDetection(img):

    # declare pointer types
    doublePtr = ct.POINTER(ct.c_double)
    doublePtrPtr = ct.POINTER(doublePtr)

    # load dll
    # dll = ct.CDLL('C:\\Users\\Dano\\PycharmProjects\\PVSO_Zadanie3\\convolution.dll')   # path to dll
    dll = ct.CDLL('C:\\Users\\Lenovo\\PycharmProjects\\PVSO_zad1\\Zadanie_3\\convolution.dll')  # path to dll
    dllFuncConvolute = dll.convolute    # function name
    dllFuncConvolute.argtypes = [doublePtrPtr, doublePtrPtr, ct.c_int, ct.c_int, ct.c_int, ct.c_int]     # inputs
    dllFuncConvolute.restype = doublePtrPtr     # output

    # convert to gray
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    img2 = gray
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # <- opencv
    # blur the image to remove noise
    # img2 = cv.GaussianBlur(gray, (3, 3), 0) # <- opencv

    # img2 = cv.GaussianBlur(img, (3, 3), 0)
    # img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # various kernels for convolution with the picture (edge detection)
    # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    # kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)  # <- good
    # kernel = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32)
    kernel = np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]], np.float64)  # <- best, from:
    # https://www.youtube.com/watch?v=uNP6ZwQ3r6A&ab_channel=FirstPrinciplesofComputerVision

    # blurring kernel
    kernel_2 = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]], np.float64)

    # apply the convolution
    hi, wi = img2.shape
    hk, wk = kernel.shape
    image_padded = np.zeros(shape=(hi + hk - 1, wi + wk - 1))
    image_padded[hk // 3:-hk // 3, wk // 3:-wk // 3] = img2
    out = np.zeros(shape=img2.shape)
    # original, VERY SLOW algorithm
    # for row in range(hi):
    #     for col in range(wi):
    #         for i in range(hk):
    #             for j in range(wk):
    #                 out[row, col] += image_padded[row + i, col + j] * kernel[i, j]

    # convert to c language types
    ct_arr_image = np.ctypeslib.as_ctypes(image_padded)
    ct_arr_kernel = np.ctypeslib.as_ctypes(kernel)
    ct_arr_kernel_2 = np.ctypeslib.as_ctypes(kernel_2)

    # fill two-dimensional pointers
    doublePtrArr = doublePtr * ct_arr_kernel._length_
    ct_arr_kernel = ct.cast(doublePtrArr(*(ct.cast(row, doublePtr) for row in ct_arr_kernel)), doublePtrPtr)
    doublePtrArr = doublePtr * ct_arr_image._length_
    ct_arr_image = ct.cast(doublePtrArr(*(ct.cast(row, doublePtr) for row in ct_arr_image)), doublePtrPtr)
    doublePtrArr = doublePtr * ct_arr_kernel_2._length_
    ct_arr_kernel_2 = ct.cast(doublePtrArr(*(ct.cast(row, doublePtr) for row in ct_arr_kernel_2)), doublePtrPtr)

    # this is used later to convert the pointer to regular matrix
    hi_temp = hi
    wi_temp = wi
    hk_temp = hk
    wk_temp = wk

    # convert to C language types
    hi = ct.c_int(hi)
    wi = ct.c_int(wi)
    hk = ct.c_int(hk)
    wk = ct.c_int(wk)

    # call the function
    # blurring
    out_ptr_blur = dllFuncConvolute(ct_arr_kernel_2, ct_arr_image, hi, wi, hk, wk)
    out_arr_blur = ptr2d_to_mat(out_ptr_blur, hi_temp, wi_temp)     # convert to matrix

    # resizing new image and converting to pointer
    image_padded[hk_temp // 3:-hk_temp // 3, wk_temp // 3:-wk_temp // 3] = out_arr_blur
    ct_arr_image = np.ctypeslib.as_ctypes(image_padded)
    doublePtrArr = doublePtr * ct_arr_image._length_
    ct_arr_image = ct.cast(doublePtrArr(*(ct.cast(row, doublePtr) for row in ct_arr_image)), doublePtrPtr)

    # edge detection
    out_ptr = dllFuncConvolute(ct_arr_kernel, ct_arr_image, hi, wi, hk, wk)
    out_mat = ptr2d_to_mat(out_ptr, hi_temp, wi_temp)

    out = np.array(out_mat, dtype=float)/float(255)  # safe conversion

    return out


def main():
    # [variables]
    # Declare the variables we are going to use
    ddepth = cv.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"
    # [variables]
    # [load]
    imageName = 'img5.jpg'
    src = cv.imread(cv.samples.findFile(imageName), cv.IMREAD_COLOR)  # Load an image
    # Check if image is loaded fine
    if src is None:
        print('Error opening image')
        return -1

    # edge detection with created algorithm
    laplace_edged_picture = laplaceEdgeDetection(src)
    cv.imshow("output", laplace_edged_picture)
    cv.waitKey()
    cv.imwrite("detected_edges.png", np.clip(laplace_edged_picture * 255, 0, 255))

    # edge detection with opencv
    # [load]
    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter
    src = cv.GaussianBlur(src, (3, 3), 0)
    # [reduce_noise]
    # [convert_to_gray]
    # Convert the image to grayscale
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # [convert_to_gray]
    # Create Window
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    # [laplacian]
    # Apply Laplace function
    dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
    # [laplacian]
    # [convert]
    # converting back to uint8
    abs_dst = cv.convertScaleAbs(dst)
    # [convert]
    # [display]
    cv.imshow(window_name, abs_dst)
    cv.waitKey(0)

    cv.imwrite("baseLaplacianOpencv.png", abs_dst)

    # [display]
    return 0

if __name__ == '__main__':
    main()

