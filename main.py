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
import torch
import torch.utils.data as data
from torch.autograd import Variable

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
from skimage import io
# Visualization
import seaborn as sns
import visdom

import os
from utils import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions, explore_spectrums, plot_spectrums,\
    sample_gt, build_dataset, show_results, compute_imf_weights
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model

import argparse

dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default=None, choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--model', type=str, default=None,
                    help="Model to train. Available:\n"
                    "SVM, SVM_grid, baseline (fully connected NN), "
                    "hu (1D CNN), "
                    "hamida (3D CNN + 1D classifier), "
                    "lee (3D FCN), "
                    "chen (3D CNN), "
                    "li (3D CNN)")
parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--cuda', type=bool, const=True, nargs='?',
                    help="Use CUDA")
parser.add_argument('--data_augmentation', type=bool, const=True, nargs='?',
                    help="Random flips (if patch_size > 1)")
parser.add_argument('--with_exploration', type=bool, const=True, nargs='?',
                    help="See data exploration visualization")
parser.add_argument('--training_sample', type=float, default=0.10,
                    help="Percentage of samples to use for training")
parser.add_argument('--epoch', type=int, help="Training epochs (optional, if"
                    " absent will be set by the model)")
parser.add_argument('--patch_size', type=int,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
parser.add_argument('--lr', type=float,
                    help="Learning rate, set by the model if not specified.")
parser.add_argument('--class_balancing', const=True, nargs='?', default=False,
                    help="Inverse median frequency class balancing (default = False)")
parser.add_argument('--sampling_mode', type=str, help="Sampling mode"
                    " (random sampling or disjoint, default: random)",
                    default='random')
parser.add_argument('--runs', type=int, default=1, help="Number of runs")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")
parser.add_argument('--train_set', type=str, default=None,
                    help="Path to the train ground truth (optional, this "
                    "supersedes the --sampling_mode option)")
parser.add_argument('--test_set', type=str, default=None,
                    help="Path to the test set (optional, by default "
                    "the test_set is the entire ground truth minus the training)")
parser.add_argument('--download', type=str, default=None, nargs='+',
                    choices=dataset_names,
                    help="Download the specified datasets and quits.")
parser.add_argument('--inference', type=str, default=None, nargs='?')
parser.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")

args = parser.parse_args()

# Use GPU ?
CUDA = args.cuda
# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
DATA_AUGMENTATION = args.data_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set

INFERENCE = args.inference
TEST_STRIDE = args.test_stride

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

viz = visdom.Visdom(env=DATASET + ' ' + MODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


if CUDA:
    print("Using CUDA")
else:
    print("Not using CUDA, will run on CPU.")

# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET,
                                                               FOLDER)
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]

# Parameters for the SVM grid search
SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
                                       'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                   {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(x):
    return convert_to_color_(x, palette=palette)
def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Instantiate the experiment based on predefined networks
kwargs = {'cuda': CUDA, 'n_classes': N_CLASSES, 'n_bands': N_BANDS,
          'epoch': EPOCH, 'ignored_labels': IGNORED_LABELS,
          'data_augmentation': DATA_AUGMENTATION, 'patch_size': PATCH_SIZE,
          'learning_rate': LEARNING_RATE, 'test_stride': TEST_STRIDE}
kwargs = dict((k, v) for k, v in kwargs.items() if v is not None)

# Show the image and the ground truth
display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, display=viz)
color_gt = convert_to_color(gt)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(img, gt, LABEL_VALUES,
                                       ignored_labels=IGNORED_LABELS,
                                       display=viz)
    plot_spectrums(mean_spectrums, display=viz)

results = []
# run the experiment several times
for run in range(N_RUNS):
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = open_file(TEST_GT)
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w,:h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
	# Sample random training spectra
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
                                                 np.count_nonzero(gt)))
    print("Running an experiment with the {} model".format(MODEL),
          "run {}/{}".format(run + 1, N_RUNS))

    if MODEL == 'SVM_grid':
        print("Running a grid search SVM")
        # Grid search SVM (linear and RBF)
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf = sklearn.model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4)
        clf.fit(X_train, y_train)
        print("SVM best parameters : {}".format(clf.best_params_))
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        save_model(clf, MODEL, DATASET)
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == 'SVM':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.svm.SVC(class_weight=class_weight)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(img.reshape(-1, N_BANDS))
        prediction = prediction.reshape(img.shape[:2])
    elif MODEL == 'SGD':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.linear_model.SGDClassifier(class_weight=class_weight, learning_rate='optimal', tol=1e-3, average=10)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(scaler.transform(img.reshape(-1, N_BANDS)))
        prediction = prediction.reshape(img.shape[:2])
    else:
        # Neural network
        model, optimizer, loss, hyperparams = get_model(MODEL, **kwargs)
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams['weights'] = torch.from_numpy(weights)
        # Split train set in train/val
        train_gt, val_gt = sample_gt(train_gt, 0.95, mode='random')
        # Generate the dataset
        train_dataset = HyperX(img, train_gt, ignored_labels=IGNORED_LABELS,
                               patch_size=hyperparams['patch_size'],
                               data_augmentation=hyperparams['data_augmentation'],
                               center_pixel=hyperparams['center_pixel'],
                               supervision=hyperparams['supervision'],
                               name=DATASET)
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=hyperparams['batch_size'],
                                       pin_memory=hyperparams['cuda'],
                                       shuffle=True)
        val_dataset = HyperX(img, val_gt, ignored_labels=IGNORED_LABELS,
                               patch_size=hyperparams['patch_size'],
                               center_pixel=hyperparams['center_pixel'])
        val_loader = data.DataLoader(val_dataset,
                                     batch_size=hyperparams['batch_size'],
                                     pin_memory=hyperparams['cuda'])

        print("Network :")
        with torch.no_grad():
            for input, _ in train_loader:
                break
            if hyperparams['cuda']:
                input = input.cuda()
            out = model(input, verbose=True)
            del(out)

        if CHECKPOINT is not None:
            model.load_state_dict(torch.load(CHECKPOINT))

        try:
            train(model, optimizer, loss, train_loader, hyperparams['epoch'],
                  scheduler=hyperparams['scheduler'], cuda=hyperparams['cuda'],
                  supervision=hyperparams['supervision'], val_loader=val_loader,
                  display=viz)
        except KeyboardInterrupt:
            # Allow the user to stop the training
            pass

        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)

    mask = np.zeros(gt.shape, dtype='bool')
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    display_predictions(color_prediction, color_gt, display=viz)

    run_results = metrics(prediction, test_gt, ignored_labels=IGNORED_LABELS, n_classes=N_CLASSES)
    results.append(run_results)
    show_results(run_results, label_values=LABEL_VALUES, display=viz)

if N_RUNS > 1:
    show_results(results, label_values=LABEL_VALUES,
                 display=viz, agregated=True)

if INFERENCE is not None:
    img = open_file(INFERENCE)[:,:,:-2]
    # Normalization
    img = np.asarray(img, dtype='float32')
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    if MODEL in ['SVM', 'SVM_grid', 'SGD']:
        from sklearn.externals import joblib
        model = joblib.load(CHECKPOINT)
        X = scaler.transform(img.reshape(-1, N_BANDS))
        prediction = model.predict(X)
        prediction = prediction.reshape(img.shape[:2])
    else:
        model = get_model(MODEL, **kwargs)[0]
        model.load_state_dict(torch.load(CHECKPOINT))
        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)

    basename = os.path.basename(INFERENCE)
    basename = str(model.__class__.__name__) + basename
    dirname = os.path.dirname(INFERENCE)
    filename = dirname + '/' + basename + '.tif'
    io.imsave(filename, prediction)
    basename = 'color_' + basename
    filename = dirname + '/' + basename + '.tif'
    io.imsave(filename, convert_to_color(prediction))
