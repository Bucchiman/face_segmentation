#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	main
# CreatedDate:  2021-09-15 04:09:09 +0900
# LastModified: 2021-09-17 18:07:45 +0900
#


import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch import distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from utils.myparse import get_args
from models.model import BiSeNet
from face_dataset import FaceMask
from mytrain import train


def main():
    # 1  left eye
    # 2  right eye
    # 3 upper lip
    # 4 lower lip
    # 5 mouth

    args = get_args()

    dist.init_process_group(backend='nccl',
                            init_method='env://')
    torch.cuda.set_device(args["local_rank"])

    resize_path = os.path.join(args["output_path"], "resize")
    mask_path = os.path.join(args["output_path"], "mask")
    roi_path = os.path.join(args["output_path"], "roi")
#    Path(resize_path).mkdir(parents=True, exist_ok=True)
#    Path(mask_path).mkdir(parents=True, exist_ok=True)
#    Path(roi_path).mkdir(parents=True, exist_ok=True)

    table = {'left_eye': 1,
             'right_eye': 2,
             'upper_lip': 3,
             'lower_lip': 4,
             'mouth': 5}

    mydataset = FaceMask(args["data_path"])
    sampler = DistributedSampler(mydataset)
    mydataloader = DataLoader(mydataset, args["batch_size"], shuffle=True, sampler=sampler)
    mymodel = BiSeNet(len(table)+1)
    mymodel = nn.parallel.DistributedDataParallel(mymodel, args["local_rank"])

    train(mydataloader, mymodel, args["device"], args["epochs"])


if __name__ == "__main__":
    main()
