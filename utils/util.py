# @Author:langyi
# @Time  :2019/4/7

import os
import torch
import pandas as pd


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # batch_size x maxk
        pred = pred.t()  # transpose, maxk x batch_size
        # target.view(1, -1): convert (batch_size,) to 1 x batch_size
        # expand_as: convert 1 x batch_size to maxk x batch_size
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # maxk x batch_size

        res = []
        for k in topk:
            # correct[:k] converts "maxk x batch_size" to "k x batch_size"
            # view(-1) converts "k x batch_size" to "(k x batch_size,)"
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_lr(epochs, optimizer):
    lr = optimizer.param_groups[0]['lr']
    lr = lr - (0.01 - 0.001) / epochs
    for _, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
