#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image
import numpy as np

from utils.mytransform import ColorJitter, HorizontalFlip, RandomCrop, RandomScale, Compose


class FaceMask(Dataset):
    def __init__(self, data_path, mode='train', cropsize=[1024, 1024],
                 *args, **kwargs):
        super(FaceMask, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        self.data_path = data_path

        self.imgs = os.listdir(os.path.join(self.data_path, 'CelebA-HQ-img'))

        #  pre-processing
        self.to_tensor = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])

        self.trans_train = Compose([ColorJitter(brightness=0.5, contrast=0.5,
                                                saturation=0.5),
                                    HorizontalFlip(),
                                    RandomScale((0.75, 1.0, 1.25,
                                                 1.5, 1.75, 2.0)),
                                    RandomCrop(cropsize)])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        impth = self.imgs[idx]
        img = Image.open(os.path.join(self.data_path, 'CelebA-HQ-img', impth))
        img = img.resize((1024, 1024), Image.BILINEAR)
        label = Image.open(os.path.join(self.data_path, 'mask', impth[:-3]+'png'))
        label = label.resize((1024, 1024), Image.BILINEAR)
        img_label = dict(img=img, label=label)
        img_label = self.trans_train(img_label)
        img, label = img_label['img'], img_label['label']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label
