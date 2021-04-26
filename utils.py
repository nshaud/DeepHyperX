# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import seaborn as sns
import itertools
import spectral
import visdom
import matplotlib.pyplot as plt
from scipy import io, misc
import os
import re
import torch
from astropy.io import fits

from PIL import Image


def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device("cuda:{}".format(ordinal))
    else:
        print(
            "/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\"
        )
        device = torch.device("cpu")
    return device


def open_file(filepath):
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    if ext == ".mat":
        # Use SciPy to load Matlab data
        return io.loadmat(filepath)
    elif ext in [".tiff", ".tif", ".jpg", ".jpeg", ".png"]:
        # Load JPG/TIFF/PNG file using Pillow
        return np.array(Image.open(filepath))
    elif ext == ".hdr":
        # Use PySpectral library to load HDR data
        img = spectral.open_image(filepath)
        return img.load()
    elif ext == ".fits":
        # Use astropy library to load data
        img = fits.open(filepath)
        return img[0].data
    else:
        raise ValueError("Unknown file format: {}".format(ext))


def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def pad_image(image, padding=None, mode="symmetric", constant=0):
    """Padding an input image.
    Modified at 2020.11.16. If you find any issues, please email at mengxue_zhang@hhu.edu.cn with details.
    Args:
        image: 2D+ image with a shape of [h, w, ...],
        The array to pad
        patch_size: optional, a list include two integers, default is [1, 1] for pure spectra algorithm,
        The patch size of the algorithm
        mode: optional, str or function, default is "symmetric",
        Including 'constant', 'reflect', 'symmetric', more details see np.pad()
        constant: optional, sequence or scalar, default is 0,
        Used in 'constant'.  The values to set the padded values for each axis
    Returns:
        padded_image with a shape of [h + patch_size[0] // 2 * 2, w + patch_size[1] // 2 * 2, ...]
    """
    if padding is None:
        return image
    h, w = padding
    pad_width = [[h, h], [w, w]] + [[0, 0] for i in image.shape[2:]]
    if mode == "constant":
        padded_image = np.pad(image, pad_width, mode=mode, constant_values=constant)
    else:
        padded_image = np.pad(image, pad_width, mode=mode)
    return padded_image


def sliding_window(image, step=(10, 10), window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(step, int):
        step = (step, step)
    w, h = window_size
    W, H = image.shape[:2]
    step_w, step_h = step
    offset_w = (W - w) % step_w
    offset_h = (H - h) % step_h
    """
    Compensate one for the stop value of range(...). because this function does not include the stop value.
    Two examples are listed as follows.
    When step = 1, supposing w = h = 3, W = H = 7, and step = 1.
    Then offset_w = 0, offset_h = 0.
    In this case, the x should have been ranged from 0 to 4 (4-6 is the last window),
    i.e., x is in range(0, 5) while W (7) - w (3) + offset_w (0) + 1 = 5. Plus one !
    Range(0, 5, 1) equals [0, 1, 2, 3, 4].

    When step = 2, supposing w = h = 3, W = H = 8, and step = 2.
    Then offset_w = 1, offset_h = 1.
    In this case, x is in [0, 2, 4] while W (8) - w (3) + offset_w (1) + 1 = 6. Plus one !
    Range(0, 6, 2) equals [0, 2, 4]/

    Same reason to H, h, offset_h, and y.
    """
    for x in range(0, W - w + offset_w + 1, step_w):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h + 1, step_h):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x : x + w, y : y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=(10, 10), window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


from datautils import IGNORED_INDEX


def metrics(prediction, target, target_names=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    # Remove ignored pixels from the metrics computation
    ignored_mask = target == IGNORED_INDEX
    target = target[~ignored_mask]
    prediction = prediction[~ignored_mask]

    if target_names is None:
        target_names = [str(i) for i in np.unique(target)]

    # Compute F1 scores and accuracy
    from sklearn.metrics import classification_report

    report = classification_report(
        target, prediction, output_dict=True, target_names=target_names
    )

    # Compute kappa coefficient
    from sklearn.metrics import cohen_kappa_score

    report["kappa"] = cohen_kappa_score(target, prediction)

    # Confusion matrix
    report["Confusion matrix"] = confusion_matrix(target, prediction)

    report["labels"] = target_names
    return report


def compute_imf_weights(ground_truth, n_classes=None):
    """ Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.

    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
        ignored_classes: id of classes to ignore (optional)
    Returns:
        numpy array with the IMF coefficients 
    """
    from datautils import IGNORED_INDEX

    valid_pixels = ground_truth != IGNORED_INDEX
    pixels = ground_truth[valid_pixels]

    n_classes = np.max(pixels) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.array([np.count_nonzero(pixels == c) for c in range(0, n_classes)], dtype="float32")
    print(frequencies)
    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.0
    print(weights)
    return weights


def camel_to_snake(name):
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s).lower()

