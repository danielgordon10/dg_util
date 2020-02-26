import multiprocessing
import time

import cv2
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
from PIL import Image
from skimage.draw import circle
from sklearn.decomposition import PCA

from . import misc_util

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO as StringIO  # Python 3.x


def tsne_image(
    features, images, img_res=64, res=4000, cval=255, point_radius=20, max_feature_size=-1, labels=None, n_threads=0
):
    """
    Embeds images via tsne into a scatter plot.

    Parameters
    ---------
    features: numpy array
        Features to visualize

    images: list or numpy array
        Corresponding images to features.

    img_res: int
        Resolution to embed images at

    res: int
        Size of embedding image in pixels

    cval: float or numpy array
        Background color value

    """
    features = np.asarray(features, dtype=np.float32)
    images = np.asarray(images)
    assert len(features.shape) == 2
    #_, uniques = misc_util.unique_rows(images.reshape(images.shape[0], -1), return_inverse=True)
    #uniques.sort()
    #images = images[uniques]
    #features = features[uniques]
    n, h, w, c = images.shape
    images = misc_util.min_resize(images.transpose(1, 2, 0, 3).reshape(h, w, n * c), img_res)
    images = images.reshape(images.shape[0], images.shape[1], n, c).transpose(2, 0, 1, 3)

    max_width = max(img_res, point_radius * 2)
    max_height = max(img_res, point_radius * 2)

    print("Starting TSNE")
    s_time = time.time()
    if 0 < max_feature_size < features.shape[-1]:
        pca = PCA(n_components=max_feature_size)
        features = pca.fit_transform(features)

    if n_threads <= 0:
        n_threads = multiprocessing.cpu_count()
    model = TSNE(n_components=2, verbose=1, random_state=0, n_jobs=n_threads)

    f2d = model.fit_transform(features)
    print("TSNE done.", (time.time() - s_time))
    print("Starting drawing.")

    xx = f2d[:, 0]
    yy = f2d[:, 1]
    if xx.max() - xx.min() > yy.max() - yy.min():
        temp = xx
        xx = yy
        yy = temp
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()
    # Fix the ratios
    sx = x_max - x_min
    sy = y_max - y_min
    if sx > sy:
        res_x = sx / float(sy) * res
        res_y = res
    else:
        res_x = res
        res_y = sy / float(sx) * res

    canvas = np.full((int(res_x + max_width), int(res_y + max_height), 3), cval, dtype=np.uint8)
    if labels is not None:
        circles = np.full(canvas.shape, 255, dtype=np.uint8)
        label_classes = np.unique(labels)
    xx = np.floor((xx - x_min) / (x_max - x_min) * res_x).astype(np.int64)
    yy = np.floor((yy - y_min) / (y_max - y_min) * res_y).astype(np.int64)
    im_ind = 0
    for x_idx, y_idx, image in zip(xx, yy, images):
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
