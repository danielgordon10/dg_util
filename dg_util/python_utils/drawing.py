from . import misc_util
import numpy as np
import cv2

BORDER = 0
CV_FONT = cv2.FONT_HERSHEY_DUPLEX
FANCY_FONT = None
FONT_SIZE = 18


def draw_contrast_text_cv2(im, text, position, size=None):
    if size is None:
        size = im.shape[1] / 320.0
    im = cv2.putText(im.copy(), text, position, CV_FONT, size, [0, 0, 0], 4)
    im = cv2.putText(im.copy(), text, position, CV_FONT, size, [255, 255, 255], 1)
    return im


def subplot(
    plots,
    rows,
    cols,
    output_width,
    output_height,
    border=BORDER,
    titles=None,
    normalize=None,
    order=None,
    fancy_text=False,
    border_color=(191, 191, 191),
):
    """
    Given a list of images, returns a single image with the sub-images tiled in a subplot.

    :param plots: array of numpy array images to plot. Can be of different sizes and dimensions as
        long as they are 2 or 3 dimensional.
    :param rows: int number of rows in subplot. If there are fewer images than rows, it will add
        empty space for the blanks. If there are fewer rows than images, it will not draw the
        remaining images.
    :param cols: int number of columns in subplot. Similar to rows.
    :param output_width: int width in pixels of a single subplot output image.
    :param output_height: int height in pixels of a single subplot output image.
    :param border: int amount of border padding pixels between each image.
    :param titles: titles for each subplot to be rendered on top of images.
    :param normalize: list of whether to subtract the max and divide by the min before colorizing.
        If none, assumed false for all images.
    :param order: if provided this reorders the provided plots before drawing them.
    :param fancy_text: if true, uses a fancier font than CV_FONT, but takes longer to render.
    :return: A single image containing the provided images (up to rows * cols).
    """
    global FANCY_FONT
    global FONT_SIZE

    if order is not None:
        plots = [plots[im_ind] for im_ind in order]
        if titles is not None:
            titles = [titles[im_ind] for im_ind in order]
        if normalize is not None:
            normalize = [normalize[im_ind] for im_ind in order]

    returned_image = np.full(
        ((output_height + border) * rows + 2 * border, (output_width + border) * cols + 2 * border, 3),
        border_color,
        dtype=np.uint8,
    )
    if fancy_text:
        from PIL import Image, ImageDraw, ImageFont

        if FANCY_FONT is None:
            FONT_SIZE = int(FONT_SIZE * output_width / 320.0)
            FANCY_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE)
    for row in range(rows):
        for col in range(cols):
            try:
                if col + cols * row >= len(plots):
                    return returned_image
                im = plots[col + cols * row]
                if im is None:
                    continue
                im = im.squeeze()
                if im.dtype != np.uint8 or len(im.shape) < 3:
                    if normalize is None or normalize[col + cols * row]:
                        im = im.astype(np.float32)
                        im -= np.min(im)
                        im *= 255 / max(np.max(im), 1e-10)
                        im = im.astype(np.uint8)
                    else:
                        im = im.astype(np.uint8)
                if len(im.shape) < 3:
                    im = cv2.applyColorMap(im, cv2.COLORMAP_JET)[:, :, ::-1]

                im = misc_util.min_side_resize_and_pad(im, output_height, output_width)
                if (
                    titles is not None
                    and len(titles) > 1
                    and len(titles) > col + cols * row
                    and len(titles[col + cols * row]) > 0
                ):
                    if fancy_text:
                        if im.dtype != np.uint8:
                            im = im.astype(np.uint8)
                        im = Image.fromarray(im)
                        draw = ImageDraw.Draw(im)
                        if isinstance(titles[col + cols * row], str):
                            for x in range(6, 9):
                                for y in range(3, 6):
                                    draw.text((x, y), titles[col + cols * row], (0, 0, 0), font=FANCY_FONT)
                            draw.text((7, 4), titles[col + cols * row], (255, 255, 255), font=FANCY_FONT)
                        else:
                            for tt, title in enumerate(titles[col + cols * row]):
                                for x in range(6, 9):
                                    for y in range(3, 6):
                                        draw.text((x, y + tt * (FONT_SIZE + 5)), title, (0, 0, 0), font=FANCY_FONT)
                                draw.text((7, 4 + tt * (FONT_SIZE + 5)), title, (255, 255, 255), font=FANCY_FONT)
                        im = np.asarray(im)
                    else:
                        scale_factor = im.shape[1] / 320.0
                        if isinstance(titles[col + cols * row], str):
                            im = draw_contrast_text_cv2(
                                im, titles[col + cols * row], (30, int(30 * scale_factor)), 0.5 * scale_factor
                            )
                        else:
                            for tt, title in enumerate(titles[col + cols * row]):
                                im = draw_contrast_text_cv2(
                                    im,
                                    titles[col + cols * row],
                                    (30, int((tt + 1) * 30 * scale_factor)),
                                    0.5 * scale_factor,
                                )
                returned_image[
                    border + (output_height + border) * row : (output_height + border) * (row + 1),
                    border + (output_width + border) * col : (output_width + border) * (col + 1),
                    :,
                ] = im
            except Exception as ex:
                import traceback

                traceback.print_exc()
                print("Failed for image", col + cols * row)
                print("shape", plots[col + cols * row].shape)
                print("type", plots[col + cols * row].dtype)
                if titles is not None and len(titles) > col + col * row:
                    print("title", titles[col + cols * row])
                import pdb

                pdb.set_trace()
                print("bad")
                raise ex

    im = returned_image
    # for one long title
    if titles is not None and len(titles) == 1:
        if fancy_text:
            if im.dtype != np.uint8:
                im = im.astype(np.uint8)
            im = Image.fromarray(im)
            draw = ImageDraw.Draw(im)
            for x in range(9, 12):
                for y in range(9, 12):
                    draw.text((x, y), titles[0], (0, 0, 0), font=FANCY_FONT)
            draw.text((10, 10), titles[0], (255, 255, 255), font=FANCY_FONT)
            im = np.asarray(im)
        else:
            scale_factor = max(max(im.shape[0], im.shape[1]) * 1.0 / 300, 1)
            cv2.putText(im, titles[0], (10, 30), CV_FONT, 0.5 * scale_factor, [0, 0, 0], 4)
            cv2.putText(im, titles[0], (10, 30), CV_FONT, 0.5 * scale_factor, [255, 255, 255], 1)

    return im


def draw_probability_hist(pi):
    p_size = max(len(pi), 100)
    p_hist = np.zeros((p_size, p_size), dtype=np.int32)
    for ii, pi_i in enumerate(pi):
        p_hist[
            : np.clip(int(np.round(pi_i * p_size)), 1, 99),
            int(ii * p_size / len(pi)) : int((ii + 1) * p_size / len(pi)),
        ] = (ii + 1)
    p_hist = np.flipud(p_hist)
    return p_hist
