# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools
import spectral
try:
    import visdom
except:
    pass
# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.legacy.nn import SpatialCrossMapLRN as SpatialCrossMapLRNOld
import re

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


def display_predictions(pred, gt, display=None):
    d_type = get_display_type(display)
    if d_type == 'visdom':
        display.images([np.transpose(pred, (2, 0, 1)),
                        np.transpose(gt, (2, 0, 1))],
                       nrow=2,
                       opts={'caption': "Prediction vs. ground truth"})
    elif d_type == 'plt':
        # Plot the results
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(pred)
        plt.title("Prediction")
        plt.axis('off')
        fig.add_subplot(1, 2, 2)
        plt.imshow(gt)
        plt.title("Ground truth")
        plt.axis('off')
        plt.show()


def display_dataset(img, gt, bands, labels, palette, display=None):
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any

    """
    d_type = get_display_type(display)
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    if d_type == 'visdom':
        caption = "RGB (bands {}, {}, {}) and ground truth".format(*bands)
        # send to visdom server
        display.images([np.transpose(rgb, (2, 0, 1)),
                        np.transpose(convert_to_color_(gt, palette=palette),
                                     (2, 0, 1))
                        ],
                       nrow=2,
                       opts={'caption': caption})
    elif d_type == 'plt':
        print("plotting with matplotlib")
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
                      ignored_labels=None, display=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        display (optional): type of display, if any
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    d_type = get_display_type(display)

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

        if d_type == 'visdom':
            pass
        elif d_type == 'plt':
            # Plot the mean spectrum with thickness based on std
            plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                             higher_spectrum, color="#3F5D7D")
            plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
            plt.show()
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums


def plot_spectrums(spectrums, display=None):
    """Plot the specified dictionary of spectrums.

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot

    """
    # Generate a color palette using seaborn
    palette = sns.color_palette("hls", len(spectrums.keys()))
    sns.set_palette(palette)

    d_type = get_display_type(display)
    if d_type == 'visdom':
        pass
    elif d_type == 'plt':
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


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results


def get_display_type(display):
    if display:
        display_type = 'plt'
        try:
            if isinstance(display, visdom.Visdom):
                display_type = 'visdom'
        except NameError:
            pass
    else:
        display_type = 'print'
    return display_type


def show_results(results, label_values=None,
                 display=None, agregated=False):
    d_type = get_display_type(display)
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        kappa = results["Kappa"]

    if d_type == 'visdom':
        display.heatmap(cm, opts={'rownames': label_values,
                                  'columnnames': label_values})
    elif d_type == 'plt':
        plt.rcParams.update({'font.size': 10})
        sns.heatmap(cm, annot=True, square=True)
        plt.title("Confusion matrix")
        plt.show()
        plt.rcParams.update({'font.size': 22})
    elif d_type == 'print':
        text += "Confusion matrix :\n"
        text += str(cm)
        text += "---\n"

    if agregated:
        text += ("Accuracy: {:.03f} +- {:.03f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.03f}%\n".format(accuracy)
    text += "---\n"

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.03f} +- {:.03f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.03f}\n".format(kappa)

    if d_type == 'visdom':
        text = text.replace('\n', '<br/>')
        display.text(text)
    else:
        print(text)


def sample_gt(gt, percentage, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    if mode == 'random':
        mask = np.zeros(gt.shape, dtype='bool')
        for l in np.unique(gt):
            x, y = np.nonzero(gt == l)
            indices = np.random.choice(len(x), int(len(x) * percentage),
                                       replace=False)
            x, y = x[indices], y[indices]
            mask[x, y] = True
        train_gt = np.zeros_like(gt)
        train_gt[mask] = gt[mask]
        test_gt = np.zeros_like(gt)
        test_gt[~mask] = gt[~mask]
    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * percentage and ratio < 1.1 * percentage:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
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
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    median = np.median(frequencies[np.nonzero(frequencies)])
    weights = median / frequencies
    weights[frequencies == 0] = 0.
    return weights

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


class SpatialCrossMapLRNFunc(Function):
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


def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()
