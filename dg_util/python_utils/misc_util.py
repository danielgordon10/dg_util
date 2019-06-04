import time
import cv2
import numpy as np


def get_time_str():
    tt = time.localtime()
    time_str = "%04d_%02d_%02d_%02d_%02d_%02d" % (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec)
    return time_str


def resize(image, width_height_tuple, interpolation=cv2.INTER_LINEAR):
    # Deal with the shitty opencv resize bug
    if image.shape[2] > 512:
        images = [image[:, :, start:min(start + 512, image.shape[2])]
                  for start in range(0, image.shape[2], 512)]
        images = [cv2.resize(image, width_height_tuple, interpolation=interpolation)
                  for image in images]
        images = [im if len(im.shape) == 3 else im[:, :, np.newaxis] for im in images]
        image = np.concatenate(images, axis=-1)
    else:
        image = cv2.resize(image, width_height_tuple, interpolation=interpolation)
    return image


def min_resize(img, size):
    """
    Resize an image so that it is size along the minimum spatial dimension.
    """
    h, w = map(float, img.shape[:2])
    if min([h, w]) != size:
        if h <= w:
            img = resize(img, (int(size), int(round((w / h) * size))))
        else:
            img = resize(img, (int(round((h / w) * size)), int(size)))
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
