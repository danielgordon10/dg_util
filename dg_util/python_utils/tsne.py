import multiprocessing
import time

import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA

from . import image_util

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO as StringIO  # Python 3.x


def tsne_image(
    features, images, img_res=64, res=4000, background_color=255,  max_feature_size=-1, labels=None, point_radius=20, n_threads=0
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

    background_color: float or numpy array
        Background color value

    max_feature_size: int
        If input_feature_size > max_feature_size> 0, features are first
        reduced using PCA to the desired size.

    point_radius: int
        Size of the circle for the label image.

    n_threads: int
        Number of threads to use for t-SNE


    labels: List or numpy array if provided
        Label for each image for drawing circle image.


    """
    features = np.asarray(features, dtype=np.float32)
    assert len(features.shape) == 2

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

    x_coords = f2d[:, 0]
    y_coords = f2d[:, 1]

    return image_util.draw_images_at_locations(images, x_coords, y_coords, img_res, res, background_color, labels, point_radius)


