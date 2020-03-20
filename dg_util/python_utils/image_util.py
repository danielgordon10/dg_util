import numbers

import cv2
import imagesize
import numpy as np
from skimage.draw import circle

from . import misc_util


# @inputImage{ndarray HxWx3} Full input image.
# @bbox{ndarray or list 4x1} bbox to be cropped in x1,y1,x2,y2 format.
# @padScale{number} scalar representing amount of padding around image.
#   padScale=1 will be exactly the bbox, padScale=2 will be 2x the input image.
# @outputSize{number} Size in pixels of output crop. Crop will be square and
#   warped.
# @return{tuple(patch, outputBox)} the output patch and bounding box
#   representing its coordinates.
def get_cropped_input(inputImage, bbox, padScale, outputSize, interpolation=cv2.INTER_LINEAR, pad_color=0):
    bbox = np.array(bbox)
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    imShape = np.array(inputImage.shape)
    if len(imShape) < 3:
        inputImage = inputImage[:, :, np.newaxis]
    xC = float(bbox[0] + bbox[2]) / 2
    yC = float(bbox[1] + bbox[3]) / 2
    boxOn = np.zeros(4)
    boxOn[0] = float(xC - padScale * width / 2)
    boxOn[1] = float(yC - padScale * height / 2)
    boxOn[2] = float(xC + padScale * width / 2)
    boxOn[3] = float(yC + padScale * height / 2)
    outputBox = boxOn.copy()
    boxOn = np.round(boxOn).astype(int)
    boxOnWH = np.array([boxOn[2] - boxOn[0], boxOn[3] - boxOn[1]])
    imagePatch = inputImage[
        max(boxOn[1], 0) : min(boxOn[3], imShape[0]), max(boxOn[0], 0) : min(boxOn[2], imShape[1]), :
    ]
    boundedBox = np.clip(boxOn, 0, imShape[[1, 0, 1, 0]])
    boundedBoxWH = np.array([boundedBox[2] - boundedBox[0], boundedBox[3] - boundedBox[1]])

    if imagePatch.shape[0] == 0 or imagePatch.shape[1] == 0:
        patch = np.zeros((int(outputSize), int(outputSize), 3), dtype=imagePatch.dtype)
    else:
        patch = cv2.resize(
            imagePatch,
            (
                max(1, int(np.round(outputSize * boundedBoxWH[0] / boxOnWH[0]))),
                max(1, int(np.round(outputSize * boundedBoxWH[1] / boxOnWH[1]))),
            ),
            interpolation=interpolation,
        )
        if len(patch.shape) < 3:
            patch = patch[:, :, np.newaxis]
        patchShape = np.array(patch.shape)

        pad = np.zeros(4, dtype=int)
        pad[:2] = np.maximum(0, -boxOn[:2] * outputSize / boxOnWH)
        pad[2:] = outputSize - (pad[:2] + patchShape[[1, 0]])

        if np.any(pad != 0):
            if len(pad[pad < 0]) > 0:
                patch = np.zeros((int(outputSize), int(outputSize), 3))
            else:
                if isinstance(pad_color, numbers.Number):
                    patch = np.pad(
                        patch, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "constant", constant_values=pad_color
                    )
                else:
                    patch = cv2.copyMakeBorder(
                        patch, pad[1], pad[3], pad[0], pad[2], cv2.BORDER_CONSTANT, value=pad_color
                    )

    return patch, outputBox


def get_image_size(fname):
    width, height = imagesize.get(fname)
    return width, height


def draw_images_at_locations(images, x_coords, y_coords, img_res=64, res=4000, background_color=255, labels=None, point_radius=20):
    """
    Plots images at the provided coordinates.

    Parameters
    ---------
    images: list or numpy array
        Images to draw.

    x_coords: list or numpy array
        x coordinates of where to draw each image.

    y_coords: list or numpy array
        y coordinates of where to draw each image.

    img_res: int
        Resolution to draw each images.

    res: int
        Size of full image in pixels.

    background_color: float or numpy array
        Background color value.

    labels: List or numpy array if provided
        Label for each image for drawing circle image.

    point_radius: int
        Size of the circle for the label image.

    """
    images = np.asarray(images)
    x_coords = np.asarray(x_coords)
    y_coords = np.asarray(y_coords)
    n, h, w, c = images.shape
    assert x_coords.shape[0] == y_coords.shape[0] == n
    images = misc_util.min_resize(images.transpose(1, 2, 0, 3).reshape(h, w, n * c), img_res)
    images = images.reshape(images.shape[0], images.shape[1], n, c).transpose(2, 0, 1, 3)

    max_width = max(img_res, point_radius * 2)
    max_height = max(img_res, point_radius * 2)
    if x_coords.max() - x_coords.min() > y_coords.max() - y_coords.min():
        temp = x_coords
        x_coords = y_coords
        y_coords = temp
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    # Fix the ratios
    sx = x_max - x_min
    sy = y_max - y_min
    if sx > sy:
        res_x = sx / float(sy) * res
        res_y = res
    else:
        res_x = res
        res_y = sy / float(sx) * res

    canvas = np.full((int(res_x + max_width), int(res_y + max_height), 3), background_color, dtype=np.uint8)
    if labels is not None:
        circles = np.full(canvas.shape, 255, dtype=np.uint8)
        label_classes = np.unique(labels)
    x_coords = np.floor((x_coords - x_min) / (x_max - x_min) * res_x).astype(np.int64)
    y_coords = np.floor((y_coords - y_min) / (y_max - y_min) * res_y).astype(np.int64)
    im_ind = 0
    for ii, x_idx, y_idx, image in zip(range(len(x_coords)), x_coords, y_coords, images):
        w, h = image.shape[:2]
        if labels is not None:
            center = (int(y_idx + h / 2.0), int(x_idx + w / 2.0))
            rr, cc = circle(center[1], center[0], point_radius)
            label = np.where(label_classes == labels[ii])[0].item()
            color = cv2.applyColorMap(
                np.array(int(label * 255.0 / len(label_classes)), dtype=np.uint8), cv2.COLORMAP_JET
            ).squeeze()
            circles[rr, cc, :] = color

        canvas[x_idx : x_idx + w, y_idx : y_idx + h] = image
        im_ind += 1

    images = list()
    images.append(canvas)

    if labels is not None:
        images.append(circles)

    return images
