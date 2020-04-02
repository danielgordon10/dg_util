import os
import pdb
import random
import time
import traceback
from typing import List, Tuple, Union

import cv2
import numpy as np
import scipy.linalg
import scipy.ndimage
import torch

from dg_util.python_utils import drawing
from dg_util.python_utils import misc_util
from dg_util.python_utils import pytorch_util as pt_util
from dg_util.python_utils import youtube_utils

DEBUG = False
SHOW_FLOW = True
LAPLACIAN_FILTER = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
LAPLACIAN_FILTER = np.tile(LAPLACIAN_FILTER[np.newaxis, np.newaxis, :, :], (3, 1, 1, 1))
LAPLACIAN_FILTER = pt_util.from_numpy(LAPLACIAN_FILTER).to(torch.float32)
EDGE_ARRAY = np.array([-1, 0, 1], dtype=np.float32)
EDGE_ARRAY = EDGE_ARRAY[:, np.newaxis]
STRUCTURE = scipy.ndimage.iterate_structure(scipy.ndimage.generate_binary_structure(2, 1), 2)
EDGE_FILTER = torch.tensor([[-1, 0, 1]], dtype=torch.float32)
MIN_SHOT_LENGTH = 10


def count_frames(path):
    video = cv2.VideoCapture(path)
    # otherwise, let's try the fast way first
    try:
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        total = -1
    video.release()
    return total


def test_speeds(video: str, sample_rate: int) -> str:
    assert sample_rate > 0
    if os.path.splitext(video)[1] == ".webm":
        return "read"
    if sample_rate < 10:
        return "read"
    if sample_rate > 50:
        return "seek"
    vidcap = cv2.VideoCapture(video)
    count = 0
    t_start = time.time()
    image = None
    # Read method
    for ii in range(sample_rate):
        success, image = vidcap.read()
        if not success:
            break
        if DEBUG and count % sample_rate == 0:
            cv2.imshow("img", image)
            cv2.waitKey(1)
        count += 1
    t_end = time.time()
    t1 = t_end - t_start
    if image is None:
        return "read"
    end_img1 = image.copy()
    if DEBUG:
        print("read total time", t_end - t_start)
    vidcap.release()

    # Seek method
    vidcap = cv2.VideoCapture(video)
    t_start = time.time()
    for ii in range(2):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, (ii * sample_rate) - 1)
        success, image = vidcap.read(1)
        if not success:
            break
        if DEBUG:
            cv2.imshow("img", image)
            cv2.waitKey(1)
        count += 1
    t_end = time.time()
    t2 = t_end - t_start
    if image is None:
        return "read"
    end_img2 = image.copy()
    if DEBUG:
        print("seek total time", t_end - t_start)
    vidcap.release()
    try:
        assert np.all(end_img1 == end_img2)
    except AssertionError:
        cv2.imshow("img1", end_img1)
        cv2.imshow("img2", end_img2)
        cv2.waitKey(0)
    if t1 < t2:
        return "read"
    else:
        return "seek"


def get_frames_by_time(video: str, start_time: int, end_time: int = -1, fps: int = 1, remove_video: bool = False):
    assert os.path.exists(video)
    vidcap = cv2.VideoCapture(video)
    try:
        vid_framerate = vidcap.get(cv2.CAP_PROP_FPS)
    except:
        # expect framerate of 30
        vid_framerate = 30
    finally:
        vidcap.release()
    start_frame = int(start_time * vid_framerate)
    end_frame = int(end_time * vid_framerate)
    sample_rate = int(vid_framerate / fps)
    return get_frames(video, sample_rate, remove_video=remove_video, start_frame=start_frame, end_frame=end_frame)


def get_frames(
    video: str,
    sample_rate: int,
    sample_method: str = None,
    remove_video: bool = False,
    max_frames: int = -1,
    start_frame: int = -1,
    end_frame: int = -1,
    return_inds=False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
    assert sample_rate > 0
    assert not ((max_frames != -1) and (end_frame != -1))
    video_start_point = 0
    if start_frame > 0:
        video_start_point = start_frame
    if max_frames > 0 and video_start_point == 0:
        num_frames_in_video = count_frames(video)
        if num_frames_in_video == -1:
            video_start_point = 0
        elif num_frames_in_video > max_frames * sample_rate:
            video_start_point = random.randint(0, num_frames_in_video - max_frames * sample_rate)
    frames = []
    frame_inds = []
    if sample_method is None:
        sample_method = test_speeds(video, sample_rate)

    if sample_method == "read":
        vidcap = cv2.VideoCapture(video)
        if video_start_point > 0:
            try:
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, video_start_point)
            except:
                vidcap.release()
                vidcap = cv2.VideoCapture(video)
                video_start_point = 0
        count = 0
        success = True
        t_start = time.time()
        total_frames = end_frame - start_frame
        frame_inds.append(video_start_point)
        curr_frame_ind = video_start_point
        while success and (max_frames < 0 or len(frames) < max_frames):
            success, image = vidcap.read()
            if not success:
                break
            if count % sample_rate == 0:
                frames.append(image[:, :, ::-1])
                frame_inds.append(curr_frame_ind)
            count += 1
            curr_frame_ind += 1
            if count >= total_frames > 0:
                break
        t_end = time.time()
        if DEBUG:
            print("total time", t_end - t_start)
        vidcap.release()

    elif sample_method == "seek":
        vidcap = cv2.VideoCapture(video)
        t_start = time.time()
        success = True
        try:
            while success and (max_frames < 0 or len(frames) < max_frames):
                frame_ind = (video_start_point + len(frames) * sample_rate) - 1
                if frame_ind > end_frame > 0:
                    break
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
                success, image = vidcap.read(1)
                if not success:
                    break
                frames.append(image[:, :, ::-1])
                frame_inds.append(frame_ind)
        except:
            frames, frame_inds = get_frames(
                video, sample_rate, "read", remove_video, max_frames, start_frame, end_frame
            )
        t_end = time.time()
        if DEBUG:
            print("total time", t_end - t_start)
        vidcap.release()

    if remove_video:
        os.remove(video)
    if return_inds:
        return frames, np.asarray(frame_inds)
    else:
        return frames


def get_subsampled_frames(
    video: str,
    sample_rate: int,
    subsample_count: int,
    subsample_rate: int,
    remove_video: bool = False,
    max_samples: int = -1,
) -> List[List[np.ndarray]]:
    assert sample_rate > 0
    video_start_point = 0
    jump = max(sample_rate, subsample_rate * subsample_count)
    num_frames_in_video = count_frames(video)
    if max_samples > 0:
        if num_frames_in_video == -1:
            video_start_point = 0
        elif num_frames_in_video > max_samples * jump:
            video_start_point = random.randint(0, num_frames_in_video - max_samples * jump)

    frames = []
    vidcap = cv2.VideoCapture(video)
    t_start = time.time()
    success = True
    try:
        while success and (max_samples < 0 or len(frames) < max_samples):
            start_frame = video_start_point + len(frames) * jump
            end_frame = start_frame + subsample_count * subsample_rate
            if end_frame > num_frames_in_video:
                break
            sub_frames = get_frames(
                video, subsample_rate, sample_method="seek", start_frame=start_frame, max_frames=subsample_count
            )
            success = len(sub_frames) == subsample_count
            if success:
                frames.append(sub_frames)
    except:
        traceback.print_exc()

    t_end = time.time()
    if DEBUG:
        print("total time", t_end - t_start)
    vidcap.release()
    if remove_video:
        os.remove(video)
    return frames


def filter_similar_frames(
    frames: List[np.ndarray], min_pix_threshold: int = 10, min_pix_percent: int = 0.1, return_inds: bool = False
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[int]]]:
    DEBUG = False
    if len(frames) < 2:
        inds = [0, 1]
        if return_inds:
            return frames, inds
        else:
            return frames
    new_frames = [frames[0]]
    inds = [0]
    last_added_image = frames[0].astype(np.int16)
    for ff in range(1, len(frames)):
        curr_frame = frames[ff].astype(np.int16)
        percent_pix_change = np.mean(np.abs(curr_frame - last_added_image) > min_pix_threshold)
        if DEBUG:
            print("pix change", percent_pix_change)
        if percent_pix_change < min_pix_percent:
            if DEBUG:
                print("percent pix change", percent_pix_change)
                cv2.imshow("img1", last_added_image.astype(np.uint8)[:, :, ::-1])
                cv2.imshow("img2", frames[ff][:, :, ::-1])
                cv2.waitKey(1)
        else:
            if DEBUG:
                print("adding")
                cv2.imshow("img1", last_added_image.astype(np.uint8)[:, :, ::-1])
                cv2.imshow("img2", frames[ff][:, :, ::-1])
                cv2.waitKey(1)
            new_frames.append(frames[ff])
            inds.append(ff)
            last_added_image = curr_frame
    if len(new_frames) * 1.0 / len(frames) < 0.35:
        if DEBUG:
            print(
                "num new frames",
                len(new_frames),
                "num_old_frames",
                len(frames),
                "ratio",
                len(new_frames) * 1.0 / len(frames),
            )
        # return [frames[0], frames[-1]]
    if len(new_frames) < 2:
        inds.append(len(frames) - 1)
        new_frames.append(frames[-1])
    if return_inds:
        return new_frames, inds
    else:
        return new_frames


def remove_border(
    images: Union[List, np.ndarray], return_inds=False
) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
    isndarray = False
    if isinstance(images, np.ndarray):
        isndarray = True
        assert len(images.shape) == 4, "Only works on TxHxWxC"
        assert images.shape[3] == 3, "Only works on TxHxWxC"
    else:
        assert len(images) > 0, "Only works on TxHxWxC"
        assert images[0].shape[2] == 3, "Only works on TxHxWxC"

    rand_inds = np.random.choice(len(images), min(10, len(images)), replace=False)
    rand_inds.sort()
    if isndarray:
        rand_images = images[rand_inds]
    else:
        rand_images = np.asarray([images[ii] for ii in rand_inds])
    masks = np.logical_or(
        np.all(rand_images < 10, axis=(0, 3)), np.all(np.equal(rand_images, rand_images[0]), axis=(0, 3))
    )
    # masks = np.all(rand_images == 0, axis=(0, 3))
    vert_masks = np.all(masks, axis=0)
    horiz_masks = np.all(masks, axis=1)
    edge_inds = []

    edge = np.where(vert_masks)[0]
    if edge.shape[0] >= 2:
        edge_min = 0
        if edge[0] == 0:
            edge_diff = edge[1:] - edge[:-1]
            edge_min = np.where(edge_diff > 1)[0]
            if len(edge_min) > 0:
                edge_min = edge_min[0] + 1  # +1 to start on good val
            else:
                edge_min = edge[-1] + 1

        edge_max = vert_masks.shape[0] - 1
        if edge[-1] == vert_masks.shape[0] - 1:
            edge_diff = edge[1:] - edge[:-1]
            edge_max = np.where(edge_diff > 1)[0]
            if len(edge_max) > 0:
                edge_max = edge[edge_max[-1] + 1]
            else:
                edge_max = edge[0]

        edge_inds.append(edge_min)
        edge_inds.append(edge_max)
        if isndarray:
            images = images[:, :, edge_min:edge_max]
        else:
            images = [image[:, edge_min:edge_max] for image in images]
    else:
        edge_inds.append(0)
        edge_inds.append(rand_images.shape[2])

    edge = np.where(horiz_masks)[0]
    if edge.shape[0] >= 2:
        edge_min = 0
        if edge[0] == 0:
            edge_diff = edge[1:] - edge[:-1]
            edge_min = np.where(edge_diff > 1)[0]
            if len(edge_min) > 0:
                edge_min = edge_min[0] + 1  # +1 to start on good val
            else:
                edge_min = edge[-1] + 1

        edge_max = horiz_masks.shape[0] - 1
        if edge[-1] == horiz_masks.shape[0] - 1:
            edge_diff = edge[1:] - edge[:-1]
            edge_max = np.where(edge_diff > 1)[0]
            if len(edge_max) > 0:
                edge_max = edge[edge_max[-1] + 1]
            else:
                edge_max = edge[0]

        if isndarray:
            images = images[:, edge_min:edge_max]
        else:
            images = [image[edge_min:edge_max] for image in images]
        edge_inds.append(edge_min)
        edge_inds.append(edge_max)
    else:
        edge_inds.append(0)
        edge_inds.append(rand_images.shape[1])

    if return_inds:
        return images, edge_inds
    else:
        return images


def filter_using_flow(
    frames0: np.ndarray, frames1: np.ndarray, return_inds=False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[int]]]:
    assert isinstance(frames0, np.ndarray)
    assert frames0.shape == frames1.shape
    original_shape = frames0.shape
    num_frames = len(frames0)
    small_frames0 = misc_util.resize(frames0, (256, 256), height_channel=1, width_channel=2)
    small_frames1 = misc_util.resize(frames1, (256, 256), height_channel=1, width_channel=2)

    small_frames0 = cv2.cvtColor(small_frames0.reshape(num_frames * 256, 256, 3), cv2.COLOR_RGB2GRAY).reshape(
        num_frames, 256, 256
    )
    small_frames1 = cv2.cvtColor(small_frames1.reshape(num_frames * 256, 256, 3), cv2.COLOR_RGB2GRAY).reshape(
        num_frames, 256, 256
    )
    masks = []

    inds = []
    large_masks = []
    if SHOW_FLOW:
        hsv = np.zeros((256, 256, 3), dtype=np.uint8)
        hsv[..., 1] = 255
    for ff in range(num_frames):
        flow = cv2.calcOpticalFlowFarneback(small_frames0[ff], small_frames1[ff], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        if SHOW_FLOW:
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow("image0", small_frames0[ff])
            cv2.imshow("image1", small_frames1[ff])
            cv2.imshow("flow", rgb)
            cv2.waitKey(0)
        mask = mag > 2
        if 0.25 < np.mean(mask) < 0.6:
            inds.append(ff)
            masks.append(mask)
            large_masks.append(
                cv2.resize(
                    mask.astype(np.uint8) * 255, (original_shape[2], original_shape[1]), interpolation=cv2.INTER_NEAREST
                )
            )
    frames = frames1[inds]

    large_masks = np.array(large_masks)
    if return_inds:
        return frames, large_masks, inds
    else:
        return frames, large_masks


def filter_using_laplacian(
    frames: Union[np.ndarray, torch.Tensor], return_inds=False
) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
    with torch.no_grad():
        assert isinstance(frames, np.ndarray) or isinstance(frames, torch.Tensor)
        if isinstance(frames, np.ndarray):
            frames_torch = pt_util.from_numpy(frames.transpose(0, 3, 1, 2)).to(torch.float32)
        else:
            frames_torch = frames
        frames_resize = torch.nn.functional.interpolate(frames_torch, (256, 256))
        laplacian = torch.nn.functional.conv2d(frames_resize, LAPLACIAN_FILTER, groups=3)
        laplacian, _ = torch.max(torch.abs(laplacian), dim=1)
        laplacian = laplacian > 3
        if DEBUG and False:
            vis_frames = pt_util.to_numpy(laplacian).astype(np.uint8) * 255
            for frame in vis_frames:
                cv2.imshow("image", frame)
                print("score", frame.mean() / 255)
                cv2.waitKey(1)
        laplacian = laplacian.to(torch.float32).mean(dim=(1, 2))
        new_frames = torch.where(laplacian > 0.1)[0]
        output = frames[new_frames]
        if len(output.shape) < 4:
            output = output[np.newaxis, ...]
        if return_inds:
            return output, new_frames
        else:
            return output


def filter_using_laplacian_opencv(
    frames: np.ndarray, return_inds=False
) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
    assert isinstance(frames, np.ndarray)
    assert len(frames.shape) == 4 and frames.shape[-1] == 3
    small_frames = frames.transpose(1, 2, 0, 3)
    small_frames = misc_util.resize(small_frames, (256, 256), height_channel=0, width_channel=1)
    small_frames = small_frames.reshape(256, 256, -1)
    small_frames_dim = small_frames.shape[-1]

    if small_frames_dim > (512 // 3) * 3:
        # Stupid opencv bug
        laplacian = [
            np.max(
                np.abs(
                    cv2.Laplacian(small_frames[:, :, start : start + (512 // 3) * 3], cv2.CV_16S).reshape(
                        256, 256, -1, 3
                    )
                ),
                axis=3,
            )
            for start in range(0, small_frames_dim, (512 // 3) * 3)
        ]
        laplacian = [(lap > 3).mean(axis=(0, 1)) for lap in laplacian]
        laplacian = np.concatenate(laplacian, axis=-1)

    else:
        laplacian = np.max(np.abs(cv2.Laplacian(small_frames, cv2.CV_16S).reshape(256, 256, -1, 3)), axis=3)
        laplacian = (laplacian > 3).mean(axis=(0, 1))
    new_frames = np.where(laplacian > 0.1)[0]
    if return_inds:
        return frames[new_frames], new_frames
    else:
        return frames[new_frames]


def get_edges(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # if len(image.shape) == 3:
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    start_t = time.time()
    if image.shape[-1] == 3:
        image = (
            np.float32(0.299) * image[..., 0] + np.float32(0.587) * image[..., 1] + np.float32(0.114) * image[..., 2]
        )
    # edge = cv2.Canny(image, 0, 50)
    image_16 = image.astype(np.int16)
    x_edge_array = EDGE_ARRAY
    y_edge_array = EDGE_ARRAY.T
    while len(x_edge_array.shape) < len(image_16.shape):
        x_edge_array = x_edge_array[np.newaxis, ...]
        y_edge_array = y_edge_array[np.newaxis, ...]

    # Some annoying opencv bug
    try:
        edge1 = cv2.filter2D(image_16, -1, x_edge_array, borderType=cv2.BORDER_REFLECT)
    except:
        edge1 = scipy.ndimage.filters.correlate(image_16, x_edge_array, mode="reflect")
    try:
        edge2 = cv2.filter2D(image_16, -1, y_edge_array, borderType=cv2.BORDER_REFLECT)
    except:
        edge2 = scipy.ndimage.filters.correlate(image_16, y_edge_array, mode="reflect")

    edge1 = np.abs(edge1)
    edge2 = np.abs(edge2)
    edge = (edge1 > 10) | (edge2 > 10)
    edge = edge * np.uint8(255)
    dilation_size = int(min(edge.shape[-2], edge.shape[-1]) * 0.01)
    if dilation_size > 1:
        dilation_kernel = np.ones((dilation_size, dilation_size), dtype=np.uint8)
        original_edge_shape = edge.shape
        while len(edge.shape) > 2:
            edge = pt_util.remove_dim(edge, 1)
        dilated = cv2.dilate(edge, dilation_kernel)
        edge = edge.reshape(original_edge_shape)
        dilated = dilated.reshape(original_edge_shape)
    else:
        dilated = edge
    inverted = 255 - dilated
    print("edges time", time.time() - start_t)
    return edge, inverted


def get_edges_pt(image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(image, torch.Tensor)
    assert len(image.shape) == 4
    # if len(image.shape) == 3:
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    start_t = time.time()
    image = image.to(torch.float32)
    image = (image * torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)).sum(1, keepdim=True)

    edge1 = torch.nn.functional.conv2d(image, EDGE_FILTER[:, np.newaxis, :, np.newaxis], padding=(1, 0)).squeeze(1)
    edge2 = torch.nn.functional.conv2d(image, EDGE_FILTER[:, np.newaxis, np.newaxis, :], padding=(0, 1)).squeeze(1)

    edge1 = torch.abs(edge1)
    edge2 = torch.abs(edge2)
    edge = (edge1 > 10) | (edge2 > 10)
    dilated = edge
    edge = pt_util.to_numpy(edge).astype(np.uint8) * 255
    dilated = pt_util.to_numpy(dilated).astype(np.uint8) * 255
    inverted = 255 - dilated
    return edge, inverted


def safe_div(x, y):
    return 0 if y == 0 else x / y


def ECR(edge1, inverted1, edge2, inverted2, crop=True):
    if crop:
        height, width = edge1.shape[:2]
        start_y = int(height * 0.3)
        end_y = int(height * 0.8)
        start_x = int(width * 0.3)
        end_x = int(width * 0.8)
        edge1 = edge1[start_y:end_y, start_x:end_x]
        edge2 = edge2[start_y:end_y, start_x:end_x]
        inverted1 = inverted1[start_y:end_y, start_x:end_x]
        inverted2 = inverted2[start_y:end_y, start_x:end_x]

    log_and1 = edge2 & inverted1
    log_and2 = edge1 & inverted2
    pixels_sum_new = np.sum(edge1)
    pixels_sum_old = np.sum(edge2)
    out_pixels = np.sum(log_and1)
    in_pixels = np.sum(log_and2)
    ecr = max(safe_div(float(in_pixels), float(pixels_sum_new)), safe_div(float(out_pixels), float(pixels_sum_old)))
    return ecr


def batch_ECR(edge1, inverted1, edge2, inverted2, crop=True):
    if crop:
        height, width = edge1.shape[1:3]
        start_y = int(height * 0.3)
        end_y = int(height * 0.8)
        start_x = int(width * 0.3)
        end_x = int(width * 0.8)
        edge1 = edge1[:, start_y:end_y, start_x:end_x]
        edge2 = edge2[:, start_y:end_y, start_x:end_x]
        inverted1 = inverted1[:, start_y:end_y, start_x:end_x]
        inverted2 = inverted2[:, start_y:end_y, start_x:end_x]

    log_and1 = edge2 & inverted1
    log_and2 = edge1 & inverted2
    pixels_sum_new = np.sum(edge1, axis=(1, 2))
    pixels_sum_old = np.sum(edge2, axis=(1, 2))
    out_pixels = np.sum(log_and1, axis=(1, 2))
    in_pixels = np.sum(log_and2, axis=(1, 2))
    ecr = np.maximum(
        in_pixels / (np.float32(1e-10) + pixels_sum_new), out_pixels / (np.float32(1e-10) + pixels_sum_old)
    )
    intersection = np.sum(~inverted1 & ~inverted2, axis=(1, 2))
    union = np.sum(edge1 | edge2, axis=(1, 2))
    iou = intersection.astype(np.float32) / union
    return np.concatenate(([0], iou))


def get_shots(
    frames: Union[np.ndarray, List[np.ndarray]], return_inds=False
) -> Union[
    List[np.ndarray],
    Tuple[List[np.ndarray], List[int]],
    List[List[np.ndarray]],
    Tuple[List[List[np.ndarray]], List[int]],
]:
    if len(frames) < 2:
        return [frames]
    last_image = frames[0]
    all_edges = None
    all_edges_inverted = None
    if isinstance(frames, np.ndarray):
        all_edges, all_edges_inverted = get_edges(frames)
    elif isinstance(frames, torch.Tensor):
        all_edges, all_edges_inverted = get_edges_pt(frames)
    if all_edges is not None:
        ecrs = batch_ECR(all_edges[:-1], all_edges_inverted[:-1], all_edges[1:], all_edges_inverted[1:], crop=False)
        threshold = np.percentile(ecrs[1:], 5)
        # Unset all the really short shots, probably false alarms.
        shots = np.where(ecrs < threshold)[0]
        shot_lengths = shots[1:] - shots[:-1]
        ecrs[shots[1:][shot_lengths < MIN_SHOT_LENGTH]] = threshold - 1e-5
        if DEBUG:
            print("threshold", threshold)
    else:
        threshold = 0.6

    if all_edges is None:
        prev_edges, prev_edges_inverted = get_edges(last_image)
    else:
        prev_edges = all_edges[0]
        prev_edges_inverted = all_edges_inverted[0]
    shot_borders = [0]
    start_t = time.time()

    for ff in range(1, len(frames) - 1):
        curr_frame = frames[ff]
        if all_edges is None:
            new_edges, new_edges_inverted = get_edges(curr_frame)
            ecr = ECR(prev_edges, prev_edges_inverted, new_edges, new_edges_inverted, crop=False)
        else:
            new_edges = all_edges[ff]
            new_edges_inverted = all_edges_inverted[ff]
            ecr = ecrs[ff]

        if DEBUG:
            if isinstance(frames, torch.Tensor):
                last_image_draw = pt_util.to_numpy(last_image).transpose(1, 2, 0)
                curr_frame_draw = pt_util.to_numpy(curr_frame).transpose(1, 2, 0)
            else:
                last_image_draw = last_image
                curr_frame_draw = curr_frame
            intersection = ~prev_edges_inverted & ~new_edges_inverted
            union = prev_edges | new_edges
            print("intersection", intersection.sum())
            print("union", union.sum())
            print("iou", intersection.sum() * 1.0 / union.sum())
            print("ecr", ecr)
            images = [
                last_image_draw,
                curr_frame_draw,
                prev_edges,
                new_edges,
                intersection,
                union,
                prev_edges & new_edges_inverted,
                new_edges & prev_edges_inverted,
            ]

            titles = ["last image", "curr image", "last edges", "curr edges", "intersection", "union"]
            print("ECR", ecr)
            image = drawing.subplot(images, 3, 2, last_image_draw.shape[1], last_image_draw.shape[0], titles=titles)
            cv2.imshow("image", image[:, :, ::-1])
            cv2.waitKey(0)

        if ecr < threshold:
            # if shot_length > 30:
            # Call this a change
            last_image = curr_frame
            if DEBUG:
                cv2.waitKey(0)
                pdb.set_trace()
            shot_borders.append(ff)
        # shot_length = 0
        prev_edges = new_edges
        prev_edges_inverted = new_edges_inverted

    shot_borders.append(len(frames))
    shot_borders = np.array(shot_borders)
    shots = []
    for ii in range(len(shot_borders) - 1):
        shots.append(frames[shot_borders[ii] : shot_borders[ii + 1]])
    if return_inds:
        return shots, shot_borders
    else:
        return shots


def example():
    video_id = "-5n-ogfW7gg"
    video = youtube_utils.download_video(video_id, "/tmp")
    # frames = get_frames('/tmp/--7qK_w-g3Y.mp4', sample_rate=5, start_frame=185 * 30, max_frames=int(10 * 30 / 6))
    # frames = get_frames_by_time("/tmp/" + video_id + ".mp4", start_time=10, end_time=15, fps=1)
    frames = get_frames(video, 1, remove_video=False, max_frames=512)
    print("num frames", len(frames))
    frames, inds = filter_similar_frames(frames, return_inds=True)
    print("num frames", len(frames))
    frames = np.stack(frames, axis=0)
    frames, inds = remove_border(frames, return_inds=True)
    frames_torch = pt_util.from_numpy(frames.transpose(0, 3, 1, 2)).to(torch.float32)
    frames_torch_resize = torch.nn.functional.interpolate(frames_torch, (256, 256))
    _, shot_borders = get_shots(frames_torch_resize, return_inds=True)
    frames_torch_resize, inds = filter_using_laplacian(frames_torch_resize, return_inds=True)
    frames = frames[inds]
    for frame in frames:
        cv2.imshow("image", frame[:, :, ::-1])
        cv2.waitKey(0)
    pdb.set_trace()
    cv2.waitKey(0)


if __name__ == "__main__":
    example()
