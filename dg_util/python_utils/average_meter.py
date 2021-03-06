from collections import deque

import torch
from .misc_util import CircularBuffer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        with torch.no_grad():
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class RollingAverageMeter(object):
    def __init__(self, window_size=10):
        super(RollingAverageMeter, self).__init__()
        self.window_size = window_size
        self.circular_buffer = None
        self.val = None
        self.avg = None
        self.total = None
        self.reset()

    def reset(self):
        self.circular_buffer = CircularBuffer(self.window_size)
        self.val = 0
        self.avg = 0
        self.total = 0

    def update(self, val):
        with torch.no_grad():
            prev_val = self.circular_buffer.append(val)
            self.val = val
            if prev_val is not None:
                self.total -= prev_val
            self.total += val
            self.avg = self.total / len(self.circular_buffer)
