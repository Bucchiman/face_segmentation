#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	parsing2colorchange
# CreatedDate:  2021-12-01 03:44:46 +0900
# LastModified: 2022-02-14 16:58:38 +0900
#


import os
from models.model import BiSeNet
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, default='../datas/one_IMG_0392_imgs_rate_30')
    parse.add_argument('--output_path', type=str)
    parse.add_argument('--model_name', type=str, default='final.pth')
    parse.add_argument('--device', default='cuda')
    parse.add_argument('--n_classes', type=int, default=10)
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
        image = img.resize((1024, 1024), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing.astype(np.uint8)


def evaluate(args):

    _, data_name = os.path.split(args["data_path"])
    table = {'left_eye': 1,
             'right_eye': 2,
             'upper_lip': 3,
             'lower_lip': 4,
             'mouth': 5,
             'left_iris': 6,
             'right_iris': 7,
             'left_pupil': 8,
             'right_pupil': 9}

    color_iris = np.array([0, 70, 255])
    color_lips = np.array([255, 79, 140])

    net = BiSeNet(n_classes=args["n_classes"])
    net = net.to(args["device"])
    net.load_state_dict(torch.load(os.path.join(args["output_path"], args["model_name"]),
                                   map_location=args["device"]))
    parsing_path = os.path.join(args["output_path"], data_name, "parsing")
    os.makedirs(parsing_path, exist_ok=True)

    for img_name in os.listdir(args["data_path"]):
        parsing = get_parsing(args["device"], args["n_classes"],
                              os.path.join(args["data_path"], img_name), net)
#        print(parsing.shape, np.unique(parsing))
#        cv2.imwrite("parsing_{}.png".format(img_name.rstrip(".jpeg")), parsing.astype(np.uint8))
#        parts = [table['left_eye'], table['right_eye'],
#                 table['upper_lip'], table['lower_lip'],
#                 table['left_iris'], table['right_iris']]
#        img = cv2.resize(cv2.imread(os.path.join(args["data_path"], img_name)), (1024, 1024))
        final_img = np.zeros((3, 1024, 1024))
        for h in range(1024):
            for w in range(1024):
                if parsing[h, w] == table["left_eye"] or parsing[h, w] == table["right_eye"]:
                    final_img[:, h, w] = 255
                elif parsing[h, w] == table["upper_lip"] or parsing[h, w] == table["lower_lip"]:
                    final_img[0, h, w] = 255
                    final_img[1, h, w] = 79
                    final_img[2, h, w] = 140
                elif parsing[h, w] == table["left_iris"] or parsing[h, w] == table["right_iris"]:
                    final_img[0, h, w] = 0
                    final_img[1, h, w] = 70
                    final_img[2, h, w] = 255
                elif parsing[h, w] == table["left_pupil"] or parsing[h, w] == table["right_pupil"]:
                    final_img[:, h, w] = 255
                else:
                    pass
#        plt.imshow(np.array(final_img).astype(np.uint8))

        cv2.imwrite("hoge.jpg", final_img.astype(np.uint8))
        break


def main():
    args = get_args()
    evaluate(args)


if __name__ == "__main__":
    main()
