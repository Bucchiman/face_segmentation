#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	preprocess
# CreatedDate:  2021-11-18 03:52:33 +0900
# LastModified: 2021-11-25 08:56:14 +0900
#


import os
import sys
import cv2
import numpy as np


def main():
    mask_path = "./cvat_datas/SegmentationClass/IMG_0272_0155.png"
    img_anno = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
    label = {"left_iris": [122, 171, 20],
             "right_iris": [255, 0, 124]}
    W, H, _ = img_anno.shape
    img_mask_left = np.zeros((W, H))
    img_mask_right = np.zeros((W, H))

    img_mask_left = np.where(img_anno == label["left_iris"], 5, 0)
    img_mask_right = np.where(img_anno == label["right_iris"], 6, 0)

    img_mask = img_mask_left + img_mask_right
    cv2.imwrite('./iris_mask/iris_mask.jpg', img_mask[..., 0].astype(np.uint8))


if __name__ == "__main__":
    main()
