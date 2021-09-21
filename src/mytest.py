#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	mytest
# CreatedDate:  2021-09-20 15:23:39 +0900
# LastModified: 2021-09-20 17:33:56 +0900
#


import os
import sys
from models.model import BiSeNet
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


def mask(image, parsing, parts):
    img_mask = np.zeros_like(image)
    for part in parts:
        img_mask = np.where(parsing == part, 255, img_mask)

    return img_mask


def roi(img_mask, img):
    img_roi = np.zeros_like(img)
    img_roi = np.where(img_mask == img_roi, img_roi, img)
    return img_roi


def get_parsing(device, n_classes, model_path, image_path, net):
    net.eval()

    to_tensor = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])

    with torch.no_grad():
        img = Image.open(image_path)
        image = img.resize((1080, 1080), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing


def evaluate(data_path, output_path, model_path, n_classes=6, device="cpu"):
    table = {'left_eye': 1,
             'right_eye': 2,
             'upper_lip': 3,
             'lower_lip': 4,
             'mouth': 5}

    net = BiSeNet(n_classes=n_classes)
    net = net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    resize_path = os.path.join(output_path, "resize")
    mask_path = os.path.join(output_path, "mask")
    roi_path = os.path.join(output_path, "roi")
    for img_name in os.listdir(data_path):
        parsing = get_parsing(device, n_classes, model_path,
                              os.path.join(data_path, img_name), net)
        parts = [table['left_eye'], table['right_eye'],
                 table['upper_lip'], table['lower_lip'],
                 table['mouth']]
        img = cv2.resize(cv2.imread(os.path.join(data_path, img_name)), (1080, 1080))
        img_bg = np.zeros((1080, 1080))

        img_mask = mask(img_bg, parsing, parts).astype(np.uint8)
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
        img_roi = roi(img_mask, img)

        cv2.imwrite(os.path.join(resize_path, img_name), img)
        cv2.imwrite(os.path.join(mask_path, img_name), img_mask)
        cv2.imwrite(os.path.join(roi_path, img_name), img_roi)


def main():
    evaluate("/Users/iwabuchi/2021/myeyes_lips_segmentation/mydatas/IMG_0272",
             "../outputs/2021_09_20_17_07_53",
             "../outputs/2021_09_20_17_07_53/final.pth")

if __name__ == "__main__":
    main()
