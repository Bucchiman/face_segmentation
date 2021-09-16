#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	myparse
# CreatedDate:  2021-09-15 04:06:05 +0900
# LastModified: 2021-09-16 17:57:20 +0900
#


import os
import sys
import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', default='/Users/iwabuchi/2021/myeyes_lips_segmentation/datas/CelebAMask-HQ')
    parse.add_argument('--output_path', default='../outputs')
    parse.add_argument('--batch_size', default=8)
    parse.add_argument('--device', default='cpu')
    args = parse.parse_args()
    return vars(args)
