#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	preprocess_datas
# CreatedDate:  2021-09-15 23:15:21 +0900
# LastModified: 2021-09-24 18:23:23 +0900
#


import os
import sys
import cv2
import numpy as np
from PIL import Image

#face_data = '../../datas/CelebAMask-HQ/CelebA-HQ-img'
#face_sep_mask = '../../datas/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
#mask_path = '../../datas/CelebAMask-HQ/mask'
face_data = '../../datas/sample_datas/CelebA-HQ-img'
face_sep_mask = '../../datas/sample_datas/CelebAMask-HQ-mask-anno'
mask_path = '../../datas/sample_datas/mask'
counter = 0
total = 0
for i in range(5):

    atts = ['l_eye', 'r_eye', 'u_lip', 'l_lip', 'l_iris', 'r_iris']

    for j in range(i*2000, (i+1)*2000):

        mask = np.zeros((512, 512))

        for l, att in enumerate(atts, 1):
            total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = os.path.join(face_sep_mask, str(i), file_name)

            if os.path.exists(path):
                counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))

                mask[sep_mask == 225] = l
        cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)


print(counter, total)
