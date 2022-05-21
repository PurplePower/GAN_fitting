import numpy as np
from PIL import Image, ImageOps


def image2dots(
        img: Image.Image, n_samples,
        canvas_width=2.0, canvas_height=2.0, std=0.02, centered=True):
    """
    Convert a gray scale image to dots.

    """

    # to gray scale and make binary
    img = ImageOps.flip(img)    # upside down
    pixels = np.array(img.convert('L'))
    pixels = pixels <= 127  # black pixels converted to 1
    points = np.argwhere(pixels)
    points = points.astype(np.float32)

    # scale to canvas and centered
    if centered:
        points[:, 0] -= img.size[0] // 2
        points[:, 1] -= img.size[1] // 2
    points[:, 0] = points[:, 0] / img.size[0] * canvas_width
    points[:, 1] = points[:, 1] / img.size[1] * canvas_height
    points[:, [0, 1]] = points[:, [1, 0]]  # swap columns

    # round to nearest n samples
    n_multiples = n_samples // points.shape[0]
    if n_samples - (n_multiples * points.shape[0]) > \
            (n_multiples + 1) * points.shape[0] - n_samples:
        n_multiples += 1

    points = np.tile(points, [n_multiples, 1])  # stack at axis 0

    # add noise
    points = points + np.random.normal(0, scale=std, size=points.shape)

    return points.astype(np.float32)
