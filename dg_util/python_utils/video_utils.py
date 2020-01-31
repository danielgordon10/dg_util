import glob
import http
import itertools
import json
import os
import pickle
import random
import re
import socket
import time
import traceback
import urllib
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pytube
import scipy.linalg
import scipy.ndimage
import torch
import tqdm
import youtube_dl
import youtube_dl.extractor.youtube as yt_extractor
from youtube_dl.compat import compat_urllib_parse
from youtube_dl.utils import orderedSet

from dg_util.python_utils import drawing
from dg_util.python_utils import misc_util
from dg_util.python_utils import pytorch_util as pt_util

DEBUG = False

USE_PYTUBE = False


def get_video_url(video_id: str) -> str:
    # url = 'https://youtu.be/%s' % video_id
    url = "https://www.youtube.com/watch?v=%s" % video_id
    # print('video', url)
    return url


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
) -> List[np.ndarray]:
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
        count = 0
        success = True
        t_start = time.time()
        total_frames = end_frame - start_frame
        while success and (max_frames < 0 or len(frames) < max_frames):
            success, image = vidcap.read()
            if not success:
                break
            if count % sample_rate == 0:
                frames.append(image[:, :, ::-1])
            count += 1
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
        except:
            frames = get_frames(video, sample_rate, "read", remove_video, max_frames, start_frame, end_frame)
        t_end = time.time()
        if DEBUG:
            print("total time", t_end - t_start)
        vidcap.release()

    if remove_video:
        os.remove(video)
    return frames


def get_subsampled_frames(
    video: str,
    sample_rate: int,
    subsample_count: int,
    subsample_rate: int,
    remove_video: bool = False,
    max_frames: int = -1,
) -> List[List[np.ndarray]]:
    assert sample_rate > 0
    video_start_point = 0
    if max_frames > 0:
        num_frames_in_video = count_frames(video)
        if num_frames_in_video == -1:
            video_start_point = 0
        elif num_frames_in_video > max_frames:
            video_start_point = random.randint(0, num_frames_in_video - max_frames)

    frames = []
    vidcap = cv2.VideoCapture(video)
    t_start = time.time()
    success = True
    try:
        while success and (max_frames < 0 or len(frames) < max_frames):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, (video_start_point + len(frames) * sample_rate) - 1)
            sub_frames = []
            for ff in range((subsample_count - 1) * subsample_rate + 1):
                success, image = vidcap.read()
                if not success:
                    break
                if ff % subsample_rate == 0:
                    sub_frames.append(image[:, :, ::-1])
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


def get_pytube_handle(video_id: str, recursion_depth: int = 0) -> Optional[pytube.YouTube]:
    try:
        with misc_util.timeout(30):
            video_url = get_video_url(video_id)
            yt = None
            streams = None
            if recursion_depth > 10:
                pass
                # print("Video", video_id, "depth", recursion_depth)
            if recursion_depth > 20:
                with open("experiment_scripts/failed_files/recursion_%s.txt" % video_id, "w") as ff:
                    ff.write("%s %s\n" % (video_id, video_url))
                    traceback.print_exc(file=ff)
                    ff.write("Recursion depth reached")
                print("Recursion depth reached", video_url)
                return None
            try:
                yt = pytube.YouTube(video_url)
            except pytube.exceptions.VideoUnavailable:
                print("Video and/or account has been removed", video_url)
                if DEBUG:
                    print("Video and/or account has been removed", video_url)
                yt = "caught"
            except KeyError as ex:
                if ex.args[0] == "url_encoded_fmt_stream_map":
                    print("Persistent problem with cipher for video", video_url)
                elif DEBUG:
                    print("Problem with cipher for video retry", video_url)
                yt = "caught"
            except (socket.gaierror, urllib.error.URLError, urllib.error.HTTPError):
                # print("Socket error, retry", video_id)
                if DEBUG:
                    print("Socket error, retry", video_id)
                yt = get_pytube_handle(video_id, recursion_depth + 1)
            except http.client.RemoteDisconnected:
                print("remote disconnect, retry", video_url)
                if DEBUG:
                    print("remote disconnect, retry", video_url)
                yt = get_pytube_handle(video_id, recursion_depth + 1)
            except (json.decoder.JSONDecodeError, TypeError, pytube.exceptions.RegexMatchError):
                print("JSON parsing error", video_url)
                with open("experiment_scripts/failed_files/json_%s.txt" % video_id, "w") as ff:
                    ff.write("%s %s\n" % (video_id, video_url))
                    traceback.print_exc(file=ff)
                    ff.write("JSONError")
                if DEBUG:
                    print("JSON parsing error", video_url)
                yt = "caught"
            except NameError as ex:
                if ex.args[0].startswith("free variable 'url'"):
                    print("Video is behind paywall", video_url)
                    if DEBUG:
                        print("Video is behind paywall", video_url)
                    yt = "caught"
            except TimeoutError:
                print("Timeout error", video_url)
                return None
            finally:
                if yt is None:
                    with open("experiment_scripts/failed_files/unknown_%s.txt" % video_id, "w") as ff:
                        ff.write("%s %s\n" % (video_id, video_url))
                        traceback.print_exc(file=ff)
                        traceback.print_exc()
                    print("Some problem with video", video_url)
                    return None
                elif yt == "caught":
                    return None

            try:
                streams = yt.streams
                assert streams is not None
            except (AttributeError, AssertionError):
                if DEBUG:
                    print("Cipher error")
                tries = 0
                while streams is None:
                    tries += 1
                    try:
                        yt = pytube.YouTube(video_url)
                        streams = yt.streams
                        assert streams is not None
                    except (AttributeError, AssertionError):
                        pass
                    if tries > 900:
                        return None
            return yt
    except TimeoutError:
        print("timeout error", video_url)
        return None


def download_video_pytube(video_id: str, video_path: str = "data/videos", recursion_depth: int = 0) -> Optional[str]:
    if not os.path.exists(video_path):
        os.makedirs(video_path, exist_ok=True)

    potential_file = glob.glob(os.path.join(video_path, video_id + "*"))
    if len(potential_file) > 0:
        if os.stat(potential_file[0]).st_size == 0:
            return None
        return potential_file[0]
    video_url = get_video_url(video_id)

    if recursion_depth > 10:
        pass
        # print("Video", video_id, "depth", recursion_depth)
    if recursion_depth > 20:
        with open("experiment_scripts/failed_files/download_recursion_depth_%s.txt" % video_id, "w") as ff:
            ff.write("%s %s\n" % (video_id, video_url))
            ff.write("Recursion depth reached")
        print("Some problem with video", video_url)
        return None

    yt = get_pytube_handle(video_id, recursion_depth)
    if yt is None:
        return None
    streams = yt.streams
    streams = (
        streams.filter(progressive=True, custom_filter_functions=[lambda x: x.resolution is not None])
        .order_by("resolution")
        .all()
    )

    if len(streams) == 0:
        print("Could not find valid videos for", video_id)
        return None
    extra_filters = [lambda x: x.mime_type == "video/mp4", lambda x: int(x.resolution[:-1]) >= 240]

    for extra_filter in extra_filters:
        if len(streams) == 1:
            break
        new_streams_list = list(filter(extra_filter, streams))
        if len(new_streams_list) > 0:
            streams = new_streams_list
    stream = streams[0]
    t_start = time.time()
    if DEBUG:
        print("downloading")
    out_video_path = os.path.join(video_path, video_id) + "." + stream.subtype
    try:
        with misc_util.timeout(30):
            filename = video_id
            while os.path.exists(os.path.join(video_path, video_id) + "." + stream.subtype):
                filename = video_id + "_" + str(int(time.time() * 100))
            out_video_path = os.path.join(video_path, video_id) + "." + stream.subtype
            video = stream.download(video_path, filename=filename)
    except (socket.gaierror, urllib.error.URLError, urllib.error.HTTPError, http.client.RemoteDisconnected) as ex:
        video = download_video_pytube(video_id, video_path, recursion_depth + 1)
    except TimeoutError:
        print("Video download timeout", video_url)
        if os.path.exists(out_video_path):
            os.remove(out_video_path)
        return None
    except:
        traceback.print_exc()
        print("Problem downloading video", video_url)
        if os.path.exists(out_video_path):
            os.remove(out_video_path)
        return None

    if video is None:
        return None
    if os.stat(video).st_size == 0:
        print("Download failed for", video)
        if os.path.exists(video):
            os.remove(video)
        return None
    if DEBUG:
        print("download done in %.3f" % (time.time() - t_start))
    return video


def ydl_download(video_id, ydl_opts):
    """
    Downloads from YDL but with error handling and shit
    :param ydl_opts:
    :return: True if success!
    """
    with misc_util.timeout(60):
        try:
            ydl = youtube_dl.YoutubeDL(ydl_opts)
            result = ydl.download([video_id])
            if result != 0:
                print("Could not download", get_video_url(video_id))
                return False
            return True
        except youtube_dl.DownloadError as e:
            if "Too Many Requests" in str(e):
                # There is no recovering from a toomanyrequests error
                raise e
                # print("sleeping bc toomanyrequests", flush=True)
                # time.sleep(random.randint(1+2*i, 5+5*i))
            else:
                print("Oh no! Problem \n\n{}\n".format(str(e)))
                return False
        except youtube_dl.utils.ExtractorError:
            print("The video is private")
            return False
        except (socket.gaierror, urllib.error.URLError, urllib.error.HTTPError):
            # print("Socket error, retry", video_id)
            if DEBUG:
                print("Socket error, retry", video_id)
        except http.client.RemoteDisconnected:
            print("remote disconnect, retry", video_id)
            if DEBUG:
                print("remote disconnect, retry", video_id)
        except (json.decoder.JSONDecodeError, TypeError, pytube.exceptions.RegexMatchError):
            print("JSON parsing error", video_id)
            with open("experiment_scripts/failed_files/json_%s.txt" % video_id, "w") as ff:
                ff.write("%s %s\n" % (video_id, video_id))
                traceback.print_exc(file=ff)
                ff.write("JSONError")
            if DEBUG:
                print("JSON parsing error", video_id)
            return False
        except TimeoutError:
            print("Timeout error", video_id)
            return False
        except Exception as e:
            traceback.print_exc()
            with open("experiment_scripts/failed_files/download_recursion_depth_%s.txt" % video_id, "w") as ff:
                ff.write("%s \n" % video_id)
                ff.write("Recursion depth reached")
            print("Some problem with video", video_id)
            print("Misc exception: {}".format(str(e)))
            return False


def download_video_ytdl_helper(id, cache_path, cookie_path=None):
    """
    Given an ID, download the video using HQ settings. (might need to turn this down, idk)

    :param id: Video id
    :param cache_path: Where to download the video
    :return: The file that we downloaded things to
    """

    ydl_opts = {
        "quiet": True,
        "cachedir": cache_path,
        "format": "best[height<=480][ext=mp4]",
        "outtmpl": os.path.join(cache_path, "%(id)s.%(ext)s"),
        "retries": 30,
        "ignoreerrors": True,
        # "source_address": "0.0.0.0",
        "socket_timeout": 30,
        "youtube_include_dash_manifest": False,
        "cookiefile": cookie_path,
        "force-ipv4": True,
    }
    if ydl_download(id, ydl_opts):
        return os.path.join(cache_path, f"{id}.mp4")
    return None


def download_video_ytdl(
    video_id: str, video_path: str = "data/videos", cookie_path: Optional[str] = None
) -> Optional[str]:
    if not os.path.exists(video_path):
        os.makedirs(video_path, exist_ok=True)

    potential_file = glob.glob(os.path.join(video_path, video_id + ".mp4"))
    if len(potential_file) > 0:
        if os.stat(potential_file[0]).st_size == 0:
            return None
        return potential_file[0]

    t_start = time.time()
    if DEBUG:
        print("downloading")
    video = download_video_ytdl_helper(video_id, video_path, cookie_path)
    if video is None or not os.path.exists(video):
        print("download error for ", video_id)
        return None
    if os.stat(video).st_size == 0:
        print("Download failed for", video)
        if os.path.exists(video):
            os.remove(video)
        return None
    if DEBUG:
        print("download done in %.3f" % (time.time() - t_start))
    return video


def download_video(video_id: str, video_path: str = "data/videos", cookie_path: Optional[str] = None) -> Optional[str]:
    if USE_PYTUBE:
        return download_video_pytube(video_id, video_path)
    else:
        return download_video_ytdl(video_id, video_path, cookie_path)


def filter_similar_frames(
    frames: List[np.ndarray], min_pix_threshold: int = 10, min_pix_percent: int = 0.1, return_inds: bool = False
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[int]]]:
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


def remove_border(images: np.ndarray, return_inds=False) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
    assert len(images.shape) == 4, "Only works on TxHxWxC"
    assert images.shape[3] == 3, "Only works on TxHxWxC"
    rand_inds = np.random.choice(images.shape[0], min(10, images.shape[0]), replace=False)
    rand_images = images[rand_inds]
    masks = np.all(rand_images < 10, axis=(0, 3))
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
        images = images[:, :, edge_min:edge_max]
    else:
        edge_inds.append(0)
        edge_inds.append(images.shape[2])

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

        images = images[:, edge_min:edge_max]
        edge_inds.append(edge_min)
        edge_inds.append(edge_max)
    else:
        edge_inds.append(0)
        edge_inds.append(images.shape[1])

    if return_inds:
        return images, edge_inds
    else:
        return images


SHOW_FLOW = True


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


LAPLACIAN_FILTER = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
LAPLACIAN_FILTER = np.tile(LAPLACIAN_FILTER[np.newaxis, np.newaxis, :, :], (3, 1, 1, 1))
LAPLACIAN_FILTER = pt_util.from_numpy(LAPLACIAN_FILTER).to(torch.float32)


def filter_using_laplacian(frames: np.ndarray, return_inds=False) -> Union[np.ndarray, Tuple[np.ndarray, List[int]]]:
    with torch.no_grad():
        assert isinstance(frames, np.ndarray) or isinstance(frames, torch.Tensor)
        if isinstance(frames, np.ndarray):
            frames_torch = pt_util.from_numpy(frames.transpose(0, 3, 1, 2)).to(torch.float32)
        else:
            frames_torch = frames
        frames_resize = torch.nn.functional.interpolate(frames_torch, (256, 256))
        laplacian = torch.nn.functional.conv2d(frames_resize, LAPLACIAN_FILTER, groups=3)
        laplacian, _ = torch.max(torch.abs(laplacian), dim=1)
        laplacian = (laplacian > 3).to(torch.float32).mean(dim=(1, 2))
        new_frames = torch.where(laplacian > 0.4)[0]
        if return_inds:
            return frames[new_frames], new_frames
        else:
            return frames[new_frames]


EDGE_ARRAY = np.array([-1, 0, 1], dtype=np.float32)
EDGE_ARRAY = EDGE_ARRAY[:, np.newaxis]
STRUCTURE = scipy.ndimage.iterate_structure(scipy.ndimage.generate_binary_structure(2, 1), 2)


def get_edges(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # if len(image.shape) == 3:
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if len(image.shape) > 2:
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
    dilation_size = int(min(edge.shape[0], edge.shape[1]) * 0.01)
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

    if all_edges is None:
        prev_edges, prev_edges_inverted = get_edges(last_image)
    else:
        prev_edges = all_edges[0]
        prev_edges_inverted = all_edges_inverted[0]
    shot_borders = [0]

    for ff in range(1, len(frames) - 1):
        curr_frame = frames[ff]
        if all_edges is None:
            new_edges, new_edges_inverted = get_edges(curr_frame)
        else:
            new_edges = all_edges[ff]
            new_edges_inverted = all_edges_inverted[ff]

        ecr = ECR(prev_edges, prev_edges_inverted, new_edges, new_edges_inverted, crop=False)
        if DEBUG:
            """
            images_old = [last_image.copy(), curr_frame.copy(),
                      prev_edges.copy(), new_edges.copy(),
                      prev_edges.copy() & new_edges_inverted.copy(), new_edges.copy() & prev_edges_inverted.copy()]
            """
            images = [
                last_image,
                curr_frame,
                prev_edges,
                new_edges,
                prev_edges & new_edges_inverted,
                new_edges & prev_edges_inverted,
            ]
            titles = ["last image", "curr image", "last edges", "curr edges", "changed edges", "changed edges"]
            print("ECR", ecr)
            image = drawing.subplot(images, 3, 2, last_image.shape[1], last_image.shape[0], titles=titles)
            cv2.imshow("image", image[:, :, ::-1])
            cv2.waitKey(1)

            """
            try:
                assert np.all([np.allclose(im1, im2) for im1, im2 in zip(images, images_old)])
            except:
                pdb.set_trace()
                print('bad')
            """

        if ecr > 0.6:
            # if shot_length > 30:
            # Call this a change
            last_image = curr_frame
            if DEBUG:
                cv2.waitKey(0)
            shot_borders.append(ff)
        # shot_length = 0
        prev_edges = new_edges
        prev_edges_inverted = new_edges_inverted

    shot_borders.append(-1)
    shots = []
    for ii in range(len(shot_borders) - 1):
        shots.append(frames[shot_borders[ii] : shot_borders[ii + 1]])
    if return_inds:
        return shots, shot_borders
    else:
        return shots


class YoutubeSearchIECC(yt_extractor.YoutubeSearchIE):
    def _get_n_results(self, query, n, search_str=""):
        """Get a specified number of results for a query"""

        videos = []
        channels = set()
        limit = n

        for pagenum in itertools.count(1):
            url_query = {"search_query": query.encode("utf-8"), "page": pagenum, "spf": "navigate"}
            url_query.update(self._EXTRA_QUERY_ARGS)
            if len(search_str) == 0:
                search_str = "CAISBhABGAEwAQ%253D%253D"  # Video, Short (<4 minutes), Creative Commons, Upload Date
            result_url = (
                "https://www.youtube.com/results?sp=" + search_str + "&" + compat_urllib_parse.urlencode(url_query)
            )
            data = self._download_json(
                result_url,
                video_id='query "%s"' % query,
                note="Downloading page %s" % pagenum,
                errnote="Unable to download API page",
            )
            html_content = data[1]["body"]["content"]

            if 'class="search-message' in html_content:
                raise Exception("[youtube] No video results", result_url)

            # video_urls = re.findall(r'href="/watch\?v=(.{11})', html_content)

            vids_and_channels = re.findall(r'href="\/watch\?v=(.{11}).+(?:user|channel)\/(.+?")', html_content)
            # print('got vids', len(videos), 'for', query)
            if len(vids_and_channels) == 0:
                break

            for vid, channel in vids_and_channels:
                if channel in channels:
                    continue
                channels.add(channel)
                videos.append(vid)

            # videos += new_videos
            if len(videos) > limit:
                break

        print("query", query, "vids", len(videos))
        if len(videos) > n:
            videos = videos[:n]
        videos = self._ids_to_results(orderedSet(videos))
        return self.playlist_result(videos, query)


def search_youtube(query, n, search_str=""):
    ydl_opts = {
        "quiet": True,
        "format": "best[height<=480][ext=mp4]",
        "retries": 20,
        "ignoreerrors": True,
        # "source_address": "0.0.0.0",
        "socket_timeout": 30,
        "youtube_include_dash_manifest": False,
        "skip_download": True,
        "writeinfojson": False,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        searcher = YoutubeSearchIECC(ydl)
        # Get results
        results = searcher._get_n_results(query, n, search_str)
        entries = results["entries"]
        ids = [entry["id"] for entry in entries]
        return ids


def example(data_path):
    subset = "val"
    with open(os.path.join(data_path, subset, "parsed_dataset_renamed.pkl"), "rb") as fi:
        dataset = pickle.load(fi)
    results = []

    # urls = sorted(glob.glob('yt_scrape/urls*.txt'), key=os.path.getmtime)
    # video_ids = [line.strip() for line in open(urls[-1])]
    # dataset = {'ids': video_ids, 'label': [0] * len(video_ids)}
    for video_id in tqdm.tqdm(dataset["ids"][:10]):
        out_folder = os.path.join("data", video_id)
        os.makedirs(out_folder, exist_ok=True)
        print("video", get_video_url(video_id))
        start_frames = time.time()
        t_start = time.time()
        video = download_video(video_id)
        print("download done in %.3f" % (time.time() - t_start))
        if video is None:
            continue
        frames = get_frames(video, 30)
        print("num frames", len(frames))
        frames = filter_similar_frames(frames)
        print("num filtered frames", len(frames))
        end_frames = time.time()
        print("ended getting frames %.3f %d" % ((end_frames - start_frames), len(frames)))
        for ff, frame in enumerate(frames):
            if DEBUG:
                cv2.imshow("img", frame)
                cv2.waitKey(1)
            cv2.imwrite(os.path.join(out_folder, "%07d.jpg" % ff), frame)


if __name__ == "__main__":
    # example('.')
    # search_youtube("apple", 100)
    video_id = "--6bJUbfpnQ"
    download_video(video_id, "/tmp")
    # frames = get_frames('/tmp/--7qK_w-g3Y.mp4', sample_rate=5, start_frame=185 * 30, max_frames=int(10 * 30 / 6))
    frames = get_frames_by_time("/tmp/" + video_id + ".mp4", start_time=10, end_time=15, fps=1)
    print("num frames", len(frames))
    import pdb

    pdb.set_trace()
    for frame in frames:
        cv2.imshow("image", frame[:, :, ::-1])
        cv2.waitKey(0)
    pdb.set_trace()
    cv2.waitKey(0)
