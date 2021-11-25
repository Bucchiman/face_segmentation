#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	myparse
# CreatedDate:  2021-09-15 04:06:05 +0900
# LastModified: 2021-09-17 17:39:57 +0900
#


import os
import sys
import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', default='../datas/CelebAMask-HQ')
    parse.add_argument('--output_path', default='../outputs')
    parse.add_argument('--config_path', default='../config')
    parse.add_argument('--batch_size', type=int, default=100)
    parse.add_argument('--device', default='cuda')
    parse.add_argument('--epochs', type=int, default=1000)
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1)
    args = parse.parse_args()
    return vars(args)
