import signal
import time

import cv2
import numpy as np


def get_time_str():
    tt = time.localtime()
    time_str = "%04d_%02d_%02d_%02d_%02d_%02d" % (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec)
    return time_str


def resize(image, width_height_tuple, interpolation=cv2.INTER_LINEAR, height_channel=0, width_channel=1):
    start_shape = image.shape
    if not (height_channel == 0 and width_channel == 1):
        # Put height and width axes first, keep everything else as it was if possible
        image = np.moveaxis(image, [height_channel, width_channel], [0, 1])
        start_shape = image.shape

    image = image.reshape(image.shape[0], image.shape[1], -1)

    # Deal with the shitty opencv resize bug
    if image.shape[2] > 512:
        images = [image[:, :, start : min(start + 512, image.shape[2])] for start in range(0, image.shape[2], 512)]
        images = [cv2.resize(image, width_height_tuple, interpolation=interpolation) for image in images]
        images = [im if len(im.shape) == 3 else im[:, :, np.newaxis] for im in images]
        image = np.concatenate(images, axis=-1)
    else:
        image = cv2.resize(image, width_height_tuple, interpolation=interpolation)

    image = image.reshape(width_height_tuple[1], width_height_tuple[0], *start_shape[2:])
    if not (height_channel == 0 and width_channel == 1):
        image = np.moveaxis(image, [0, 1], [height_channel, width_channel])
    return image


def min_resize(img, size, interpolation=cv2.INTER_LINEAR):
    """
    Resize an image so that it is size along the minimum spatial dimension.
    """
    h, w = map(float, img.shape[:2])
    if min([h, w]) != size:
        if h <= w:
            img = resize(img, (int(round((w / h) * size)), int(size)), interpolation)
        else:
            img = resize(img, (int(size), int(round((h / w) * size))), interpolation)
    return img


def max_resize(img, size, interpolation=cv2.INTER_LINEAR):
    """
    Resize an image so that it is size along the maximum spatial dimension.
    """
    h, w = map(float, img.shape[:2])
    if max([h, w]) != size:
        if h >= w:
            img = resize(img, (int(round((w / h) * size)), int(size)), interpolation)
        else:
            img = resize(img, (int(size), int(round((h / w) * size))), interpolation)
    return img


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
        im = resize(im, (im_width, im_height), interpolation=interpolation)
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


def unique_rows(arr, return_index=False, return_inverse=False):
    arr = arr.copy()
    b = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    inv = None
    if return_inverse:
        _, idx, inv = np.unique(b, return_index=True, return_inverse=True)
    else:
        _, idx = np.unique(b, return_index=True)
    unique = arr[idx]
    if return_index and return_inverse:
        return unique, idx, inv
    elif return_index:
        return unique, idx
    elif return_inverse:
        return unique, inv
    else:
        return unique


class timeout(object):
    # Taken from https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
