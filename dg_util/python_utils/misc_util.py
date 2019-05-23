import time
import cv2
import numpy as np


def get_time_str():
    tt = time.localtime()
    time_str = "%04d_%02d_%02d_%02d_%02d_%02d" % (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec)
    return time_str


def min_side_resize_and_pad(im, output_height, output_width, interpolation=cv2.INTER_NEAREST):
    if im.shape[:2] != (output_height, output_width):
        im_width = im.shape[1] * output_height / im.shape[0]
        if im_width > output_width:
            im_width = output_width
            im_height = im.shape[0] * output_width / im.shape[1]
        else:
            im_width = im.shape[1] * output_height / im.shape[0]
            im_height = output_height
        im_width = int(im_width)
        im_height = int(im_height)
        im = cv2.resize(im, (im_width, im_height), interpolation=interpolation)
        if im_width != output_width:
            pad0 = int(np.floor((output_width - im_width) * 1.0 / 2))
            pad1 = int(np.ceil((output_width - im_width) * 1.0 / 2))
            if len(im.shape) == 3:
                im = np.lib.pad(im, ((0, 0), (pad0, pad1), (0, 0)), "constant", constant_values=0)
            else:
                im = np.lib.pad(im, ((0, 0), (pad0, pad1)), "constant", constant_values=0)
        elif im_height != output_height:
            pad0 = int(np.floor((output_height - im_height) * 1.0 / 2))
            pad1 = int(np.ceil((output_height - im_height) * 1.0 / 2))
            if len(im.shape) == 3:
                im = np.lib.pad(im, ((pad0, pad1), (0, 0), (0, 0)), "constant", constant_values=0)
            else:
                im = np.lib.pad(im, ((pad0, pad1), (0, 0)), "constant", constant_values=0)
    return im
