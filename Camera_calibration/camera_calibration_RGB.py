# -*- coding: utf-8 -*-
"""
Calibrate the Camera with Zhang Zhengyou Method.
image with distortion
"""

import os

from calibration_helper import Calibrator
import pyzed.sl as sl
import cv2
import numpy as np


def main():
    img_dir = "./image/right"
    shape_inner_corner = (11, 8)
    size_grid = 0.02
    # create calibrator
    calibrator = Calibrator(img_dir, shape_inner_corner, size_grid)
    # calibrate the camera
    mat_intri, coff_dis = calibrator.calibrate_camera()



    # 
    # 
    image = cv2.imread('./image/right/img14.jpg')
    height, width = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_intri, coff_dis, (width, height), 1, (width, height))
    image_un = cv2.undistort(image, mat_intri, coff_dis, None, newcameramtx)
    while True:
        cv2.imshow('undistorted image', image_un)
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            cv2.destroyAllWindows()
            break
        



if __name__ == '__main__':
    main()