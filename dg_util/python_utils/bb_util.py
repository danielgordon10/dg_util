import numbers

import numpy as np

LIMIT = 99999999


# BBoxes are [x1, y1, x2, y2]
def clip_bbox(bboxes, min_clip, max_x_clip, max_y_clip):
    bboxes_out = bboxes
    added_axis = False
    if len(bboxes_out.shape) == 1:
        added_axis = True
        bboxes_out = bboxes_out[:, np.newaxis]
    bboxes_out[[0, 2]] = np.clip(bboxes_out[[0, 2]], min_clip, max_x_clip)
    bboxes_out[[1, 3]] = np.clip(bboxes_out[[1, 3]], min_clip, max_y_clip)
    if added_axis:
        bboxes_out = bboxes_out[:, 0]
    return bboxes_out


# [x1 y1, x2, y2] to [xMid, yMid, width, height]
def xyxy_to_xywh(bboxes, clip_min=-LIMIT, clip_width=LIMIT, clip_height=LIMIT, round=False):
    added_axis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        added_axis = True
        bboxes = bboxes[:, np.newaxis]
    bboxes_out = np.zeros(bboxes.shape)
    x1 = bboxes[0]
    y1 = bboxes[1]
    x2 = bboxes[2]
    y2 = bboxes[3]
    bboxes_out[0] = (x1 + x2) / 2.0
    bboxes_out[1] = (y1 + y2) / 2.0
    bboxes_out[2] = x2 - x1
    bboxes_out[3] = y2 - y1
    if clip_min != -LIMIT or clip_width != LIMIT or clip_height != LIMIT:
        bboxes_out = clip_bbox(bboxes_out, clip_min, clip_width, clip_height)
    if bboxes_out.shape[0] > 4:
        bboxes_out[4:] = bboxes[4:]
    if added_axis:
        bboxes_out = bboxes_out[:, 0]
    if round:
        bboxes_out = np.round(bboxes_out).astype(int)
    return bboxes_out


# [xMid, yMid, width, height] to [x1 y1, x2, y2]
def xywh_to_xyxy(bboxes, clip_min=-LIMIT, clip_width=LIMIT, clip_height=LIMIT, round=False):
    added_axis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        added_axis = True
        bboxes = bboxes[:, np.newaxis]
    bboxes_out = np.zeros(bboxes.shape)
    xMid = bboxes[0]
    yMid = bboxes[1]
    width = bboxes[2]
    height = bboxes[3]
    bboxes_out[0] = xMid - width / 2.0
    bboxes_out[1] = yMid - height / 2.0
    bboxes_out[2] = xMid + width / 2.0
    bboxes_out[3] = yMid + height / 2.0
    if clip_min != -LIMIT or clip_width != LIMIT or clip_height != LIMIT:
        bboxes_out = clip_bbox(bboxes_out, clip_min, clip_width, clip_height)
    if bboxes_out.shape[0] > 4:
        bboxes_out[4:] = bboxes[4:]
    if added_axis:
        bboxes_out = bboxes_out[:, 0]
    if round:
        bboxes_out = np.round(bboxes_out).astype(int)
    return bboxes_out


# @bboxes {np.array} 4xn array of boxes to be scaled
# @scalars{number or arraylike} scalars for width and height of boxes
# @in_place{bool} If false, creates new bboxes.
def scale_bbox(bboxes, scalars, clip_min=-LIMIT, clip_width=LIMIT, clip_height=LIMIT, round=False, in_place=False):
    added_axis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes, dtype=np.float32)
    if len(bboxes.shape) == 1:
        added_axis = True
        bboxes = bboxes[:, np.newaxis]
    if isinstance(scalars, numbers.Number):
        scalars = np.full((2, bboxes.shape[1]), scalars, dtype=np.float32)
    if not isinstance(scalars, np.ndarray):
        scalars = np.array(scalars, dtype=np.float32)
    if len(scalars.shape) == 1:
        scalars = np.tile(scalars[:, np.newaxis], (1, bboxes.shape[1])).astype(np.float32)

    bboxes = bboxes.astype(np.float32)

    width = bboxes[2] - bboxes[0]
    height = bboxes[3] - bboxes[1]
    x_mid = (bboxes[0] + bboxes[2]) / 2.0
    y_mid = (bboxes[1] + bboxes[3]) / 2.0
    if not in_place:
        bboxes_out = bboxes.copy()
    else:
        bboxes_out = bboxes

    bboxes_out[0] = x_mid - width * scalars[0] / 2.0
    bboxes_out[1] = y_mid - height * scalars[1] / 2.0
    bboxes_out[2] = x_mid + width * scalars[0] / 2.0
    bboxes_out[3] = y_mid + height * scalars[1] / 2.0

    if clip_min != -LIMIT or clip_width != LIMIT or clip_height != LIMIT:
        bboxes_out = clip_bbox(bboxes_out, clip_min, clip_width, clip_height)
    if added_axis:
        bboxes_out = bboxes_out[:, 0]
    if round:
        bboxes_out = np.round(bboxes_out).astype(np.int32)
    return bboxes_out


def make_square(bboxes, in_place=False):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        num_boxes = 1
        width = bboxes[2] - bboxes[0]
        height = bboxes[3] - bboxes[1]
    else:
        num_boxes = bboxes.shape[1]
        width = bboxes[2] - bboxes[0]
        height = bboxes[3] - bboxes[1]
    max_size = np.maximum(width, height)
    scalars = np.zeros((2, num_boxes))
    scalars[0] = max_size * 1.0 / width
    scalars[1] = max_size * 1.0 / height
    return scale_bbox(bboxes, scalars, in_place=in_place)


# Converts from the full image coordinate system to range 0:crop_padding. Useful for getting the coordinates
#   of a bounding box from image coordinates to the location within the cropped image.
# @bbox_to_change xyxy bbox whose coordinates will be converted to the new reference frame
# @crop_location xyxy box of the new origin and max points (without padding)
# @crop_padding the amount to pad the crop_location box (1 would be keep it the same, 2 would be doubled)
# @crop_size the maximum size of the coordinate frame of bbox_to_change.
def to_crop_coordinate_system(bbox_to_change, crop_location, crop_padding, crop_size):
    if isinstance(bbox_to_change, list):
        bbox_to_change = np.array(bbox_to_change)
    if isinstance(crop_location, list):
        crop_location = np.array(crop_location)
    bbox_to_change = bbox_to_change.astype(np.float32)
    crop_location = crop_location.astype(np.float32)

    crop_location = scale_bbox(crop_location, crop_padding)
    crop_location_xywh = xyxy_to_xywh(crop_location)
    bbox_to_change -= crop_location[[0, 1, 0, 1]]
    bbox_to_change *= crop_size / crop_location_xywh[[2, 3, 2, 3]]
    return bbox_to_change


# Inverts the transformation from to_crop_coordinate_system
# @crop_size the maximum size of the coordinate frame of bbox_to_change.
def from_crop_coordinate_system(bbox_to_change, crop_location, crop_padding, crop_size):
    if isinstance(bbox_to_change, list):
        bbox_to_change = np.array(bbox_to_change)
    if isinstance(crop_location, list):
        crop_location = np.array(crop_location)
    bbox_to_change = bbox_to_change.astype(np.float32)
    crop_location = crop_location.astype(np.float32)

    crop_location = scale_bbox(crop_location, crop_padding)
    crop_location_xywh = xyxy_to_xywh(crop_location)
    bbox_to_change *= crop_location_xywh[[2, 3, 2, 3]] / crop_size
    bbox_to_change += crop_location[[0, 1, 0, 1]]
    return bbox_to_change


"""
@rects1 numpy dx4 matrix of bounding boxes
@rect2 single numpy 1x4 matrix of bounding box
@return dx1 IOUs
Rectangles are [x1, y1, x2, y2]
"""


def IOU_numpy(rects1, rect2):
    # intersection = np.fmin(np.zeros((rects1.shape[0],1))
    (d, n) = rects1.shape
    x1s = np.fmax(rects1[:, 0], rect2[0])
    x2s = np.fmin(rects1[:, 2], rect2[2])
    y1s = np.fmax(rects1[:, 1], rect2[1])
    y2s = np.fmin(rects1[:, 3], rect2[3])
    ws = np.fmax(x2s - x1s, 0)
    hs = np.fmax(y2s - y1s, 0)
    intersection = ws * hs
    rects1Area = (rects1[:, 2] - rects1[:, 0]) * (rects1[:, 3] - rects1[:, 1])
    rect2Area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union = np.fmax(rects1Area + rect2Area - intersection, 0.00001)
    return intersection * 1.0 / union


def IOU_lists(rects1, rects2):
    (d, n) = rects1.shape
    x1s = np.fmax(rects1[:, 0], rects2[:, 0])
    x2s = np.fmin(rects1[:, 2], rects2[:, 2])
    y1s = np.fmax(rects1[:, 1], rects2[:, 1])
    y2s = np.fmin(rects1[:, 3], rects2[:, 3])
    ws = np.fmax(x2s - x1s, 0)
    hs = np.fmax(y2s - y1s, 0)
    intersection = ws * hs
    rects1Area = (rects1[:, 2] - rects1[:, 0]) * (rects1[:, 3] - rects1[:, 1])
    rects2Area = (rects2[:, 2] - rects2[:, 0]) * (rects2[:, 3] - rects2[:, 1])
    union = np.fmax(rects1Area + rects2Area - intersection, 0.00001)
    return intersection * 1.0 / union


# Rectangles are [x1, y1, x2, y2]
def IOU(rect1, rect2):
    if not isinstance(rect1, np.ndarray):
        rect1 = np.array(rect1)
    if not isinstance(rect2, np.ndarray):
        rect2 = np.array(rect2)
    rect1 = [min(rect1[[0, 2]]), min(rect1[[1, 3]]), max(rect1[[0, 2]]), max(rect1[[1, 3]])]
    rect2 = [min(rect2[[0, 2]]), min(rect2[[1, 3]]), max(rect2[[0, 2]]), max(rect2[[1, 3]])]
    intersection = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])) * max(
        0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])
    )

    union = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]) + (rect2[2] - rect2[0]) * (rect2[3] - rect2[1]) - intersection

    return intersection * 1.0 / max(union, 0.00001)


def intersection(rect1, rect2):
    return max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])) * max(
        0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])
    )


"""
@rects1 numpy dx5 matrix of bounding boxes
@rect2 single numpy 1x4 matrix of bounding box
@return nx5 rects where n is number of rects over overlapThresh
Rectangles are [x1, y1, x2, y2, 0]
"""


def get_overlapping_boxes(rects1, rect2, overlapThresh=0.001):
    x1s = np.fmax(rects1[:, 0], rect2[0])
    x2s = np.fmin(rects1[:, 2], rect2[2])
    y1s = np.fmax(rects1[:, 1], rect2[1])
    y2s = np.fmin(rects1[:, 3], rect2[3])
    ws = np.fmax(x2s - x1s, 0)
    hs = np.fmax(y2s - y1s, 0)
    intersection = ws * hs
    rects1Area = (rects1[:, 2] - rects1[:, 0]) * (rects1[:, 3] - rects1[:, 1])
    rect2Area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union = np.fmax(rects1Area + rect2Area - intersection, 0.00001)
    ious = intersection * 1.0 / union
    rects1[:, 4] = ious
    rects1 = rects1[ious > overlapThresh, :]
    return rects1


"""
@rects1 numpy dx4 matrix of bounding boxes
@rect2 single numpy 1x4 matrix of bounding box
@return number of rects over overlapThresh
Rectangles are [x1, y1, x2, y2]
"""


def count_overlapping_boxes(rects1, rect2, overlapThresh=0.001):
    if rects1.shape[1] == 0:
        return 0
    x1s = np.fmax(rects1[:, 0], rect2[0])
    x2s = np.fmin(rects1[:, 2], rect2[2])
    y1s = np.fmax(rects1[:, 1], rect2[1])
    y2s = np.fmin(rects1[:, 3], rect2[3])
    ws = np.fmax(x2s - x1s, 0)
    hs = np.fmax(y2s - y1s, 0)
    intersection = ws * hs
    rects1Area = (rects1[:, 2] - rects1[:, 0]) * (rects1[:, 3] - rects1[:, 1])
    rect2Area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union = np.fmax(rects1Area + rect2Area - intersection, 0.00001)
    ious = intersection * 1.0 / union
    return np.sum(ious > overlapThresh)
