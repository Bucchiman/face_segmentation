#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	iris_plus
# CreatedDate:  2021-11-18 04:51:41 +0900
# LastModified: 2021-11-27 06:45:51 +0900
#


import os
import sys
import cv2
import numpy as np


def main():
    one_mask = cv2.imread("./original_datas/eyes_lips_mask.jpg", 0)
    two_mask = cv2.resize(cv2.imread("iris_mask/iris_mask.jpg", 0), (1024, 1024))
    mask = one_mask.copy()
    for i in range(two_mask.shape[0]):
        for j in range(two_mask.shape[1]):
            if two_mask[i, j] == 6:
                mask[i, j] = 6
            elif two_mask[i, j] == 7:
                mask[i, j] = 7
    cv2.imwrite("iris_mask/eyes_lips_iris_mask.jpg", mask.astype(np.uint8))


if __name__ == "__main__":
    main()
