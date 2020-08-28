import logging as log

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from webcam_cv3_dlib2_api import debug


def grayplt(img):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    if np.size(img.shape) == 3:
        ax.imshow(img, vmin=0, vmax=1)
    else:
        ax.imshow(img, cmap='hot', vmin=0, vmax=1)
    plt.show()


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def preprocess_image(img):
    imag = cv2.imread(img)
    res = cv2.resize(imag, (160, 160), interpolation=cv2.INTER_CUBIC)
    res = np.expand_dims(res, axis=0)
    return res


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)

    b = np.sum(np.multiply(source_representation, source_representation))

    c = np.sum(np.multiply(test_representation, test_representation))

    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation

    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))

    euclidean_distance = np.sqrt(euclidean_distance)

    # euclidean_distance = l2_normalize(euclidean_distance )

    return euclidean_distance


def image_process2(detector, imag, gamma):
    if debug == 1: grayplt(imag / 255)
    imag = adjust_gamma(imag, gamma)
    if debug == 1: grayplt(imag / 255)
    result = detector.detect_faces(imag)
    if debug == 1: log.deug("rawimage {}".format(result))
    if result == []: return False, imag
    keypoints = result[0]['keypoints']
    turned = 0

    while keypoints['right_eye'][1] - keypoints['left_eye'][1] > 8:
        imag2 = ndimage.rotate(imag, 2, mode='nearest')
        if debug == 1: log.deug("turned")
        turned = 1
        result2 = detector.detect_faces(imag2)
        if result2 == []: break
        imag = imag2
        result = result2
        keypoints = result[0]['keypoints']

    while keypoints['left_eye'][1] - keypoints['right_eye'][1] > 8:
        imag2 = ndimage.rotate(imag, -2, mode='nearest')
        if debug == 1: log.deug("turned")
        turned = 1
        result2 = detector.detect_faces(imag2)
        if result2 == []: break
        imag = imag2
        result = result2
        keypoints = result[0]['keypoints']

    if turned == 1:
        if debug == 1: grayplt(imag / 255)

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.

    bounding_box = result[0]['box']

    if debug == 1: log.deug("bounding_box {}".format(bounding_box))
    if bounding_box[3] < 45: return False, imag
    if bounding_box[2] < 45: return False, imag

    if debug == 1: log.deug("keypoints {}".format(keypoints))

    if keypoints == {}: return False, imag

    if 'left_eye' not in keypoints: return False, imag
    if 'right_eye' not in keypoints: return False, imag
    if 'mouth_left' not in keypoints: return False, imag
    if 'mouth_right' not in keypoints: return False, imag
    if 'nose' not in keypoints: return False, imag

    if result[0]['confidence'] < 0.95: return False, imag

    left_bound = int(bounding_box[0])  # +(keypoints['left_eye'][0]-bounding_box[0])/3 )
    right_bound = int(
        bounding_box[0] + bounding_box[2])  # -(bounding_box[0]+bounding_box[2]-keypoints['right_eye'][0])/3 )
    top_bound = int(bounding_box[1])  # +(min(keypoints['right_eye'][1],keypoints['left_eye'][1])-bounding_box[1])/3 )
    bottom_bound = int(bounding_box[1] + bounding_box[
        3])  # -(bounding_box[1]+bounding_box[3]-max(keypoints['mouth_right'][1],keypoints['mouth_left'][1]))/3 )

    left_length = keypoints['nose'][0] - left_bound
    right_length = right_bound - keypoints['nose'][0]
    top_length = keypoints['nose'][1] - top_bound
    bottom_length = bottom_bound - keypoints['nose'][1]
    imag = imag[top_bound:bottom_bound, left_bound:right_bound]

    if debug == 1: grayplt(imag / 255)

    imag = (imag - imag % 16)
    # continue
    return True, imag
