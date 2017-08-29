"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep models (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""
# -*- coding: utf-8 -*-
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch.utils.data as data
from torch.autograd import Variable

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection

try:
    import visdom
    # Open connection to Visdom server
    viz = visdom.Visdom()
except ImportError:
    # Fallback on Matplotlib + Seaborn
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (8, 8)
    import seaborn as sns

from utils import metrics, convert_to_color_, convert_from_color_,\
                  display_dataset, explore_spectrums, plot_spectrums,\
                  sample_gt, build_dataset
from datasets import get_dataset, HyperX
from models import get_model, train, test

import argparse

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, required=True,
                    help="Dataset to use. Available:\n"
                    "Pavia Center (PaviaC), Pavia University (PaviaU), "
                    "Kennedy Space Center (KSC), Indian Pines (IndianPines), "
                    "Botswana")
parser.add_argument('--model', type=str, required=True,
                    help="Model to train. Available:\n"
                    "SVM, baseline (fully connected NN), "
                    "hamida (3D CNN + 1D classifier), "
                    "lee (3D FCN), "
                    "chen (3D CNN), "
                    "li (3D CNN)")
parser.add_argument('--folder', type=str, help="Path to dataset folder.",
                    default="./")
parser.add_argument('--cuda', type=bool, default=False, help="Use CUDA")
parser.add_argument('--with_exploration', type=bool, default=False,
                    help="See data exploration visualization")
parser.add_argument('--training_sample', type=float, default=0.05,
                    help="Percentage of samples to use for training")
parser.add_argument('--epoch', type=int, help="Training epochs (optional, if"
                    " absent will be set by the model)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs")

args = parser.parse_args()

# Use GPU ?
CUDA = args.cuda
# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
DATASET = args.dataset
MODEL = args.model
N_RUNS = args.runs
DATAVIZ = args.with_exploration
FOLDER = args.folder

# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS = get_dataset(DATASET,
                                                               FOLDER)
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]

# Parameters for the SVM grid search
SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                                       'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]

# Generate color palette
palette = {0: (0, 0, 0)}
for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
    palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Show the image and the ground truth
display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, visdom=viz)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(img, gt, LABEL_VALUES,
                                       ignored_labels=IGNORED_LABELS,
                                       visdom=viz)
    plot_spectrums(mean_spectrums, visdom=viz)

# run the experiment several times
for run in range(N_RUNS):
    # Sample random training spectra
    train_gt = sample_gt(gt, SAMPLE_PERCENTAGE)
    color_gt = convert_to_color(train_gt)
    print("{} samples randomly selected".format(np.count_nonzero(gt)))

    if MODEL == 'SVM':
        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)

        clf = sklearn.svm.SVC()
        clf = sklearn.model_selection.GridSearchCV(clf, SVM_GRID_PARAMS)
        clf.fit(X_train, y_train)
        print("SVM best parameters : {}".format(clf.best_params_))

        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])

    else:
        print("Running an experiment with the {} model".format(MODEL))
        # Instantiate the experiment based on predefined networks
        model, optimizer, loss, hyperparams = get_model(MODEL, cuda=CUDA,
                                                        n_classes=N_CLASSES,
                                                        n_bands=N_BANDS)

        # Generate the dataset
        train_dataset = HyperX(img, train_gt, ignored_labels=IGNORED_LABELS,
                               patch_size=hyperparams['patch_size'],
                               data_augmentation=False,
                               center_pixel=hyperparams['center_pixel'])
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=hyperparams['batch_size'],
                                       pin_memory=hyperparams['cuda'],
                                       shuffle=True)

        print("Network :")
        for input, _ in train_loader:
            break
        if hyperparams['cuda']:
            input = input.cuda()
        data = Variable(input, volatile=True)
        out = model(data, verbose=True)
        del(out)

        train(model, optimizer, loss, train_loader, hyperparams['epoch'],
              cuda=hyperparams['cuda'], visdom=viz)

        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)

    if viz:
        viz.images([np.transpose(convert_to_color(prediction), (2, 0, 1)),
                    np.transpose(convert_to_color(gt), (2, 0, 1))],
                   nrow=2)
    else:
        # Plot the results
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(convert_to_color(prediction))
        plt.title("Prediction")
        plt.axis('off')
        fig.add_subplot(1, 2, 2)
        plt.imshow(convert_to_color(gt))
        plt.title("Ground truth")
        plt.axis('off')
        plt.show()

    metrics(prediction, gt, ignored_labels=IGNORED_LABELS,
            label_values=LABEL_VALUES)
