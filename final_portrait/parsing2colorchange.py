#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#FileName: 	parsing2colorchange
# CreatedDate:  2021-12-02 19:20:29 +0000
# LastModified: 2021-12-02 20:39:22 +0000
#


import os
import sys
import numpy as np
import cv2
import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, default='outputs/2021_11_29_02_18_22/resize')
    parse.add_argument('--root_path', type=str, default='outputs/2021_11_29_02_18_22')
    args = parse.parse_args()
    return vars(args)



def main():
    args = get_args()
    root_path = args["root_path"]
    final_imgs_path = os.path.join(root_path, "final_imgs")
    os.makedirs(final_imgs_path, exist_ok=True)

    table = {'left_eye': 1,
             'right_eye': 2,
             'upper_lip': 3,
             'lower_lip': 4,
             'left_iris': 6,
             'right_iris': 7}

    for img_name in os.listdir(args["data_path"]):
        print(img_name)
        img = cv2.cvtColor(cv2.imread(os.path.join(args["data_path"], img_name)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(root_path, "parsing", "parsing_"+img_name.rstrip(".jpeg")+".png"), 0)
        color = np.zeros_like(img)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == table['left_eye'] or mask[i, j] == table['right_eye']:
                    color[i, j] = 255
                elif mask[i, j] == table['left_iris'] or mask[i, j] == table['right_iris']:
                    if img[i, j, 0] > 80 and img[i, j, 1] > 80 and img[i, j, 2] > 80:
                        color[i, j] = 255
                    else:
                        color[i, j, 0] = 0
                        color[i, j, 1] = 70
                        color[i, j, 2] = 255
                elif mask[i, j] == table['upper_lip'] or mask[i, j] == table['lower_lip']:
                    color[i, j, 0] = 255
                    color[i, j, 1] = 79
                    color[i, j, 2] = 140
                else:
                    pass
        cv2.imwrite(os.path.join(final_imgs_path, img_name), cv2.cvtColor(color.astype(np.uint8), cv2.COLOR_RGB2BGR))





if __name__ == "__main__":
    main()
