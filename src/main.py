#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	main
# CreatedDate:  2021-09-15 04:09:09 +0900
# LastModified: 2021-09-19 14:55:35 +0900
#


import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from datetime import datetime
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
    nowtime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = os.path.join(args["output_path"], nowtime)
    Path(output_dir).mkdir(exist_ok=False, parents=True)

    torch.cuda.set_device(args["local_rank"])
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:33241',
                            world_size=torch.cuda.device_count(),
                            rank=args["local_rank"])
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
    mydataloader = DataLoader(mydataset, args["batch_size"], shuffle=False, sampler=sampler)
    mymodel = BiSeNet(len(table)+1)
    mymodel.to(args["device"])
    mymodel = nn.parallel.DistributedDataParallel(mymodel, device_ids=[args["local_rank"]],
                                                  output_device=args["local_rank"])

    train(output_dir, mydataloader, mymodel, args["epochs"])


if __name__ == "__main__":
    main()
