#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	mytrain
# CreatedDate:  2021-09-16 17:28:20 +0900
# LastModified: 2021-09-17 02:28:10 +0900
#


import os
import sys
import torch
import time
import datetime
from utils.loss import OhemCELoss
from utils.optimizer import Optimizer


def train(dataloader, net, device, epochs):
    net = net.to(device)
    net.train()
    ignore_idx = -100
    score_thres = 0.7
    n_min = 1024*1024
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    # optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optim = Optimizer(model=net, lr0=lr_start, momentum=momentum,
                      wd=weight_decay, warmup_steps=warmup_steps,
                      warmup_start_lr=warmup_start_lr, max_iter=epochs,
                      power=power)

    # train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    epoch = 0
    for it in range(epochs):
        for im, lb in dataloader:
            im = im.to(device)
            lb = lb.to(device)
            H, W = im.size()[2:]
            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            out, out16, out32 = net(im)

            lossp = LossP(out, lb)
            loss2 = Loss2(out16, lb)
            loss3 = Loss3(out32, lb)
            loss = lossp + loss2 + loss3
            loss.backward()
            optim.step()

            loss_avg.append(loss.item())

            #  print training log message
            if (it+1) % msg_iter == 0:
                loss_avg = sum(loss_avg) / len(loss_avg)
                lr = optim.lr
                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                eta = int((epochs - it) * (glob_t_intv / it))
                eta = str(datetime.timedelta(seconds=eta))
                msg = ', '.join(['it: {it}/{max_it}', 'lr: {lr:4f}',
                                 'loss: {loss:.4f}', 'eta: {eta}',
                                 'time: {time:.4f}']).format(it=it+1,
                                                             max_it=epochs,
                                                             lr=lr,
                                                             loss=loss_avg,
                                                             time=t_intv,
                                                             eta=eta)

                loss_avg = []
                st = ed
    torch.save(net.to('cpu').state_dict(), "final.pth")
