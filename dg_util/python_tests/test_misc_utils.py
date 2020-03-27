
import numpy as np
from dg_util.python_utils import misc_util
import cv2


def _test_min_resize(input_shape, expected_shape, size, height_channel=0, width_channel=1, always_resize=True):
    input = np.zeros(input_shape, dtype=np.uint8)
    output = misc_util.min_resize(input, size, cv2.INTER_LINEAR, height_channel, width_channel, always_resize)
    assert output.shape == expected_shape


def _test_max_resize(input_shape, expected_shape, size, height_channel=0, width_channel=1, always_resize=True):
    input = np.zeros(input_shape, dtype=np.uint8)
    output = misc_util.max_resize(input, size, cv2.INTER_LINEAR, height_channel, width_channel, always_resize)
    assert output.shape == expected_shape


def test_min_resize():
    input_shape = (480, 640, 3)
    output_shape = (240, 320, 3)
    _test_min_resize(input_shape, output_shape, 240)

    input_shape = (480, 640, 3)
    _test_max_resize(input_shape, input_shape, 240, always_resize=False)


def test_max_resize():
    input_shape = (480, 640, 3)
    output_shape = (240, 320, 3)
    _test_max_resize(input_shape, output_shape, 320)

    input_shape = (480, 640, 3)
    _test_max_resize(input_shape, input_shape, 640, always_resize=False)


def _test_center_crop(input_shape, expected_shape, aspect_ratio):
    input = np.zeros(input_shape, dtype=np.uint8)
    output = misc_util.center_crop(input, aspect_ratio)
    assert output.shape == expected_shape


def test_center_crop():
    input_shape = (200, 200, 3)
    output_shape = (200, 200, 3)
    _test_center_crop(input_shape, output_shape, 1)

    input_shape = (200, 200, 3)
    output_shape = (200, 100, 3)
    _test_center_crop(input_shape, output_shape, 0.5)

    input_shape = (200, 200, 3)
    output_shape = (100, 200, 3)
    _test_center_crop(input_shape, output_shape, 2)

    input_shape = (200, 300, 3)
    output_shape = (150, 300, 3)
    _test_center_crop(input_shape, output_shape, 2)

    input_shape = (300, 200, 3)
    output_shape = (300, 150, 3)
    _test_center_crop(input_shape, output_shape, 0.5)



