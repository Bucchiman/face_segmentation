#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	mytest
# CreatedDate:  2021-09-20 15:23:39 +0900
# LastModified: 2021-09-24 18:21:45 +0900
#


import os
from pathlib import Path
from models.model import BiSeNet
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, default='../datas/IMG_0272')
    parse.add_argument('--output_path', type=str)
    parse.add_argument('--model_name', type=str, default='final.pth')
    parse.add_argument('--device', default='cuda')
    parse.add_argument('--n_classes', type=int, default=6)
    args = parse.parse_args()
    return vars(args)


def mask(image, parsing, parts):
    img_mask = np.zeros_like(image)
    for part in parts:
        img_mask = np.where(parsing == part, 255, img_mask)

    return img_mask


def roi(img_mask, img):
    img_roi = np.zeros_like(img)
    img_roi = np.where(img_mask == img_roi, img_roi, img)
    return img_roi


def get_parsing(device, n_classes, image_path, net):
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


def evaluate(args):
    table = {'left_eye': 1,
             'right_eye': 2,
             'upper_lip': 3,
             'lower_lip': 4,
             'mouth': 5}

    net = BiSeNet(n_classes=args["n_classes"])
    net = net.to(args["device"])
    net.load_state_dict(torch.load(os.path.join(args["output_path"], args["model_name"]),
                                   map_location=args["device"]))
    resize_path = os.path.join(args["output_path"], "resize")
    mask_path = os.path.join(args["output_path"], "mask")
    roi_path = os.path.join(args["output_path"], "roi")
    Path(resize_path).mkdir(parents=True, exist_ok=True)
    Path(mask_path).mkdir(parents=True, exist_ok=True)
    Path(roi_path).mkdir(parents=True, exist_ok=True)

    for img_name in os.listdir(args["data_path"]):
        parsing = get_parsing(args["device"], args["n_classes"],
                              os.path.join(args["data_path"], img_name), net)
        parts = [table['left_eye'], table['right_eye'],
                 table['upper_lip'], table['lower_lip'],
                 table['mouth']]
        img = cv2.resize(cv2.imread(os.path.join(args["data_path"], img_name)), (1024, 1024))
        img_bg = np.zeros((1024, 1024))

        img_mask = mask(img_bg, parsing, parts).astype(np.uint8)
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
        img_roi = roi(img_mask, img)

        cv2.imwrite(os.path.join(resize_path, img_name), img)
        cv2.imwrite(os.path.join(mask_path, img_name), img_mask)
        cv2.imwrite(os.path.join(roi_path, img_name), img_roi)


def main():
    args = get_args()
    evaluate(args)


if __name__ == "__main__":
    main()
