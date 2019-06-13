import os
import tempfile
import time

import cv2
import numpy as np
import scipy.misc
import scipy.misc
import tensorflow as tf
from skimage.draw import circle
from sklearn.manifold import TSNE
from torchviz import make_dot

from . import misc_util

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO as StringIO  # Python 3.x


def kernel_to_image(data, padsize=1):
    """Turns a convolutional kernel into an image of nicely tiled filters.
    :param data: numpy array in format N x C x H x W.
    :param padsize: optional int to indicate visual padding between the filters.
    :return: image of the filters in a tiled/mosaic layout
    """
    if len(data.shape) > 4:
        data = np.squeeze(data)
    data = np.transpose(data, (0, 2, 3, 1))
    data_shape = tuple(data.shape)
    min_val = np.min(np.reshape(data, (data_shape[0], -1)), axis=1)
    data = np.transpose((np.transpose(data, (1, 2, 3, 0)) - min_val), (3, 0, 1, 2))
    max_val = np.max(np.reshape(data, (data_shape[0], -1)), axis=1)
    data = np.transpose((np.transpose(data, (1, 2, 3, 0)) / max_val), (3, 0, 1, 2))

    n = int(np.ceil(np.sqrt(data_shape[0])))
    ndim = len(data.shape)
    padding = ((0, n ** 2 - data_shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (ndim - 3)
    data = np.pad(data, padding, mode="constant", constant_values=0)
    # tile the filters into an image
    data_shape = data.shape
    data = np.transpose(np.reshape(data, ((n, n) + data_shape[1:])), ((0, 2, 1, 3) + tuple(range(4, ndim + 1))))
    data_shape = data.shape
    data = np.reshape(data, ((n * data_shape[1], n * data_shape[3]) + data_shape[4:]))
    return (data * 255).astype(np.uint8)


class SummaryWriter(tf.summary.FileWriter):
    def __init__(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        super(SummaryWriter, self).__init__(path)
        import threading

        self.lock = threading.Lock()
        self.count = 0

    def add_summary(self, summary, global_step=None, increment_step_counter=True):
        self.lock.acquire()
        if global_step is None:
            global_step = 0
        if self.count < global_step:
            self.count = global_step
        elif increment_step_counter:
            self.count += 1
        super(SummaryWriter, self).add_summary(summary, self.count)
        self.flush()
        self.lock.release()

    def increment(self):
        self.lock.acquire()
        self.count += 1
        self.lock.release()


# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    @property
    def count(self):
        return self.writer.count

    @count.setter
    def count(self, new_count):
        if self.writer.count < new_count:
            self.writer.count = new_count

    def multi_scalar_log(self, tags, values, step):
        for tag, value in zip(tags, values):
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step, False)
        self.writer.increment()

    def dict_log(self, items_to_log, step):
        tags, values = zip(*items_to_log.items())
        self.multi_scalar_log(tags, values, step)

    def scalar_summary(self, tag, value, step, increment_counter):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step, increment_counter)

    def network_graph_summary(self, final_layer, named_parameters, step):
        dot = make_dot(final_layer, named_parameters)
        path = tempfile.mktemp(".gv")
        dot.render(path, format="png")
        image = scipy.misc.imread(path + ".png")[:, :, :3]
        self.image_summary("network", image, step, False)

    def network_conv_summary(self, network, step):
        for ii, (name, val) in enumerate(network.state_dict().items()):
            val = val.detach().cpu().numpy()
            name = "layer_%03d/" % ii + name
            if len(val.squeeze().shape) == 4:
                self.conv_variable_summaries(val, step, name, False)
            else:
                self.variable_summaries(val, step, name, False)
        self.writer.increment()

    def network_variable_summary(self, network, step):
        for ii, (name, val) in enumerate(network.state_dict().items()):
            name = "layer_%03d/" % ii + name
            val = val.detach().cpu().numpy()
            self.variable_summaries(val, step, name, False)
        self.writer.increment()

    def variable_summaries(self, var, step, scope="", increment_counter=True):
        # Some useful stats for variables.
        if len(scope) > 0:
            scope = "/" + scope
        scope = "summaries" + scope
        mean = np.mean(np.abs(var))
        self.scalar_summary(scope + "/mean_abs", mean, step, increment_counter)

    def conv_variable_summaries(self, var, step, scope="", increment_counter=True):
        # Useful stats for variables and the kernel images.
        self.variable_summaries(var, step, scope, increment_counter)
        if len(scope) > 0:
            scope = "/" + scope
        scope = "conv_summaries" + scope + "/filters"
        var_shape = var.shape
        if not (var_shape[0] == 1 and var_shape[1] == 1):
            if var_shape[2] < 3:
                var = np.tile(var, [1, 1, 3, 1])
                var_shape = var.shape
            summary_image = kernel_to_image(var[:, :3, :, :])[np.newaxis, ...]
            self.image_summary(scope, summary_image, step, increment_counter)

    def image_summary(self, tag, images, step, increment_counter, max_size=1000):
        """Log a list of images."""

        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                images = [images]
            elif len(images.shape) == 2:
                images = [np.tile(images[:, :, np.newaxis], (1, 1, 3))]
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            if max(img.shape[:2]) > max_size:
                img = misc_util.max_resize(img, max_size, interpolation=cv2.INTER_NEAREST)
            s = StringIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=img.shape[0], width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag="%s/%d" % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step, increment_counter)

    def histo_summary(self, tag, values, step, bins=1000, increment_counter=True):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step, increment_counter)
        self.writer.flush()

    def tsne_summary(self, tag, features, images, step, increment_counter=True, img_res=64, res=4000, cval=255, labels=None):
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
        assert len(features.shape) == 2
        RADIUS = 20
        features = np.array(features, dtype=np.float64)
        _, uniques = misc_util.unique_rows(images.reshape(images.shape[0], -1), return_inverse=True)
        uniques.sort()
        images = images[uniques]
        features = features[uniques]
        n, h, w, c = images.shape
        images = misc_util.min_resize(images.transpose(1, 2, 0, 3).reshape(h, w, n * c), img_res)
        images = images.reshape(images.shape[0], images.shape[1], n, c).transpose(2, 0, 1, 3)

        max_width = max(img_res, RADIUS * 2)
        max_height = max(img_res, RADIUS * 2)

        print("Starting TSNE")
        s_time = time.time()
        model = TSNE(n_components=2, verbose=1, random_state=0)
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
        x_coords = np.linspace(x_min, x_max, res_x)
        y_coords = np.linspace(y_min, y_max, res_y)
        im_ind = 0
        inds = list(range(len(images)))
        for ii, x, y, image in zip(inds, xx, yy, images):
            w, h = image.shape[:2]
            x_idx = np.argmin((x - x_coords) ** 2)
            y_idx = np.argmin((y - y_coords) ** 2)
            if labels is not None:
                center = (int(y_idx + h / 2.0), int(x_idx + w / 2.0))
                rr, cc = circle(center[1], center[0], RADIUS)
                import pdb
                pdb.set_trace()
                label = np.where(label_classes == labels[ii])[0]
                color = cv2.applyColorMap(
                    np.array(int(label * 255.0 / len(label_classes)), dtype=np.uint8), cv2.COLORMAP_JET
                ).squeeze()
                circles[rr, cc, :] = color

            canvas[x_idx : x_idx + w, y_idx : y_idx + h] = image
            im_ind += 1

        img_summaries = []
        # Write the image to a string
        s = StringIO()
        scipy.misc.toimage(canvas).save(s, format="jpeg")
        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=canvas.shape[0], width=canvas.shape[1])
        # Create a Summary value
        img_summaries.append(tf.Summary.Value(tag=tag, image=img_sum))

        if labels is not None:
            s = StringIO()
            scipy.misc.toimage(circles).save(s, format="jpeg")
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=circles.shape[0], width=circles.shape[1])
            img_summaries.append(tf.Summary.Value(tag="%s_labels" % tag, image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step, increment_counter)
