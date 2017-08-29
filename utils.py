# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools
import spectral
# Torch
import torch
import torch.nn as nn
from torch.autograd import Function as F
from torch.legacy.nn import SpatialCrossMapLRN as SpatialCrossMapLRNOld


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


def display_dataset(img, gt, bands, labels, palette, visdom=None):
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        visdom (optional): visdom instance to connect (fallback to matplotlib)

    """
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    # TODO : fixme
    if visdom:
        # send to visdom server
        visdom.images([np.transpose(rgb, (2, 0, 1)),
                       np.transpose(convert_to_color_(gt, palette=palette),
                                    (2, 0, 1))
                       ],
                      nrow=2)
    else:
        # use Matplotlib
        fig = plt.figure()
        fig.add_subplot(121)
        plt.imshow(rgb)
        plt.title("RGB composite image (bands {}, {}, {})".format(*bands))
        plt.axis('off')
        fig.add_subplot(122)
        # Display the colorized ground truth
        plt.imshow(convert_to_color_(gt, palette=palette))
        for color, label in zip(palette.values(), labels):
            color = tuple(np.asarray(color, dtype='float') / 255)
            plt.plot(0, 0, "-", c=color, label=label)
        plt.legend(labels, loc=9, bbox_to_anchor=(0.5, 0.), ncol=3)
        plt.title("Ground truth")
        plt.axis('off')
        plt.show()


def explore_spectrums(img, complete_gt, class_names,
                      ignored_labels=None, visdom=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore

    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}

    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        if visdom:
            step = max(1, class_spectrums.shape[0] // 100)
            # Sample and plot spectrums from the selected class
            for spectrum in class_spectrums[::step, :]:
                plt.title(class_names[c])
                plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        if visdom:
            # Plot the mean spectrum with thickness based on std
            plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                             higher_spectrum, color="#3F5D7D")
            plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
            plt.show()
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums


def plot_spectrums(spectrums, visdom=None):
    """Plot the specified dictionary of spectrums.

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot

    """
    # Generate a color palette using seaborn
    palette = sns.color_palette("hls", len(spectrums.keys()))
    sns.set_palette(palette)

    # TODO : fix me
    if visdom:
        fig = plt.figure()
        for k, v in spectrums.items():
            plt.plot(v)
            n_bands = len(v)
        axes = fig.axes[0]
        axes.set_xlim(0, n_bands)
        axes.set_ylim(0,)
        plt.title('Mean spectra')
        # Place the legend under the plot
        plt.legend(spectrums.keys(), loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.show()


def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def get_random_pos(img, window_shape):
    """ Return the corners of a random window in the input image

    Args:
        img: 2D (or more) image, e.g. RGB or grayscale image
        window_shape: (width, height) tuple of the window

    Returns:
        xmin, xmax, ymin, ymax: tuple of the corners of the window

    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
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
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
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


def metrics(prediction, target, ignored_labels=None, label_values=None,
            details=True, visual=False):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        visual (optional): bool set to True to use seaborn plots

    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    cm = confusion_matrix(
        target,
        prediction)

    if details and visual:
        plt.rcParams.update({'font.size': 10})
        sns.heatmap(cm, annot=True, square=True)
        plt.title("Confusion matrix")
        plt.show()
        plt.rcParams.update({'font.size': 22})
    elif details:
        print("Confusion matrix :")
        print(cm)
        print("---")

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    total = sum(sum(cm))
    if details:
        print("{} pixels processed".format(total))
        print("Total accuracy : {:.4f}%".format(accuracy))

        print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    if details:
        print("F1Score :")
        for l_id, score in enumerate(F1Score):
            print("\t{}: {:.4f}".format(label_values[l_id], score))

        print("---")

        # Compute kappa coefficient
        pa = np.trace(cm) / float(total)
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
            float(total * total)
        kappa = (pa - pe) / (1 - pe)
        print("Kappa: {:.4f}".format(kappa))
    return accuracy


def sample_gt(gt, percentage):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        gt_out: a 2D array of int labels with zeroes on removed labels

    """
    gt_out = np.copy(gt)
    mask = np.random.rand(*gt.shape) > percentage
    gt_out[mask] = 0
    return gt_out


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """2D version of the PyTorch cross entropy loss.

    Args:
        input: PyTorch tensor of predictions
        target: PyTorch tensor of labels
        weight: PyTorch tensor of weights for the classes
        size_average (optional): bool, set to True to average the loss on the
        tensor
    Returns:
        cross entropy loss
    """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

# LRN2D for PyTorch, from :
# https://github.com/pytorch/pytorch/issues/653#issuecomment-304361386

# function interface, internal, do not use this one!!!


class SpatialCrossMapLRNFunc(F):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        self.save_for_backward(input)
        self.lrn = SpatialCrossMapLRNOld(
            self.size, self.alpha, self.beta, self.k)
        self.lrn.type(input.type())
        return self.lrn.forward(input)

    def backward(self, grad_output):
        input, = self.saved_tensors
        return self.lrn.backward(input, grad_output)

# use this one instead


class SpatialCrossMapLRN(nn.Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(SpatialCrossMapLRN, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return SpatialCrossMapLRNFunc(self.size, self.alpha,
                                      self.beta, self.k)(input)
