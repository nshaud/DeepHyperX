# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep models (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection

# Visualization
import seaborn as sns

import os
from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    compute_imf_weights,
    get_device,
)
from visualization import (
    explore_spectrums,
    plot_spectrums,
    display_dataset,
    display_predictions,
    show_results,
)
from sampling import split_ground_truth
from datasets import get_dataset, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model
from utils import open_file

import argparse


def main(args):
    CUDA_DEVICE = get_device(args.cuda)
    N_JOBS = args.n_jobs
    # % of training samples
    SAMPLE_PERCENTAGE = args.training_sample
    # Data augmentation ?
    FLIP_AUGMENTATION = args.flip_augmentation
    RADIATION_AUGMENTATION = args.radiation_augmentation
    MIXTURE_AUGMENTATION = args.mixture_augmentation
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
    # Maximum number of blocks for "blocks" sampling_mode, e.g 8
    N_BLOCKS = args.nblocks
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
    TRAIN_OVERLAP = args.train_overlap
    TEST_OVERLAP = args.test_overlap

    if args.download is not None and len(args.download) > 0:
        for dataset in args.download:
            get_dataset(dataset, target_folder=FOLDER)
        quit()

    hyperparams = vars(args)
    # Load the dataset
    img, gt, LABEL_VALUES, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)
    IGNORED_LABELS = [0]
    # Number of classes
    N_CLASSES = len(LABEL_VALUES)
    print('N_CLASSES =', len(LABEL_VALUES))
    # Number of bands (last dimension of the image tensor)
    if isinstance(img,list):
        N_BANDS = img[0][0].data.shape[-1]
        print('N_BANDS = ',img[0][0].data.shape[-1])
    else:
        N_BANDS = img.shape[-1]

    if palette is None:
        # Generate color palette
        palette = {0: (0, 0, 0)}
        for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
    invert_palette = {v: k for k, v in palette.items()}

    def convert_to_color(x):
        return convert_to_color_(x, palette=palette)

    def convert_from_color(x):
        return convert_from_color_(x, palette=invert_palette)

    # Instantiate the experiment based on predefined networks
    hyperparams.update(
        {
            "n_classes": N_CLASSES,
            "n_bands": N_BANDS,
            "ignored_labels": IGNORED_LABELS,
            "device": CUDA_DEVICE,
        }
    )
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    # Show the image and the ground truth
    writer = SummaryWriter(comment=f"-{DATASET}_{MODEL}")
    if isinstance(img,list):
        display_dataset(img[0][0].data,gt[0][0].data, 
                        RGB_BANDS, LABEL_VALUES, palette, writer=writer)
    else:
        display_dataset(img,gt, RGB_BANDS, LABEL_VALUES, palette, writer=writer)
        
    #color_gt = convert_to_color(gt)

    if DATAVIZ:
        # Data exploration : compute and show the mean spectrums
        mean_spectrums = explore_spectrums(
            img, gt, LABEL_VALUES, writer, ignored_labels=IGNORED_LABELS
        )
        plot_spectrums(mean_spectrums, writer, title="Mean spectrum/class")

    results = []
    # run the experiment several times
    for run in range(N_RUNS):
        if isinstance(img,list):
            print(
                "Running an experiment with the {} model".format(MODEL),
                "run {}/{}".format(run + 1, N_RUNS),
            )
            
            display_predictions(convert_to_color(gt[0][0].data), 
                                writer, caption="Ground Truth")
            
            from datautils import MultiDataset
            
            train_dataset = MultiDataset(
                img, gt, window_size=hyperparams["patch_size"]
            )
            print("Sample in dataset: {}".format(len(train_dataset)))
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=hyperparams["batch_size"],
                # pin_memory=hyperparams['device'],
                shuffle=True,
                num_workers=N_JOBS,
            )
            # val_dataset = HyperX(img, val_gt, **hyperparams)
            val_dataset = MultiDataset(
                img, gt, window_size=hyperparams["patch_size"]
            )
            val_loader = data.DataLoader(
                val_dataset,
                # pin_memory=hyperparams['device'],
                batch_size=hyperparams["batch_size"],
                num_workers=N_JOBS,
            )
        else:
            if TRAIN_GT is not None and TEST_GT is not None:
                train_gt = open_file(TRAIN_GT).astype("uint8")
                test_gt = open_file(TEST_GT).astype("uint8")
            elif TRAIN_GT is not None:
                train_gt = open_file(TRAIN_GT).astype("uint8")
                test_gt = np.copy(gt)
                w, h = test_gt.shape
                test_gt[(train_gt > 0)[:w, :h]] = 0
            elif TEST_GT is not None:
                test_gt = open_file(TEST_GT)
            else:
                # Sample random training spectra
                print(SAMPLE_PERCENTAGE)
                print(gt.shape)
                train_gt, test_gt = split_ground_truth(
                    gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE, nblocks=N_BLOCKS
                )
            print(train_gt)
            from datautils import count_valid_pixels

            print(
                "{} samples selected (over {})".format(
                    count_valid_pixels(train_gt), count_valid_pixels(gt)
                )
            )
            print(
                "Running an experiment with the {} model".format(MODEL),
                "run {}/{}".format(run + 1, N_RUNS),
            )

            display_predictions(convert_to_color(train_gt), writer, caption="Train GT")
            display_predictions(convert_to_color(test_gt), writer, caption="Test GT")

            from shallow_models import SKLEARN_MODELS
            from shallow_models import fit_sklearn_model
            from shallow_models import infer_from_sklearn_model

            if MODEL in SKLEARN_MODELS:
                from datautils import to_sklearn_datasets

                X_train, y_train = to_sklearn_datasets(img, train_gt)
                clf = fit_sklearn_model(
                    MODEL,
                    X_train,
                    y_train,
                    exp_name=DATASET,
                    class_balancing=CLASS_BALANCING,
                    n_jobs=N_JOBS,
                )
                prediction = infer_from_sklearn_model(clf, img)
            else:
                if CLASS_BALANCING:
                    weights = compute_imf_weights(train_gt, n_classes=N_CLASSES)
                    hyperparams["weights"] = torch.from_numpy(weights).float()
                # TODO: Split train set in train/val
                # train_gt, val_gt = split_ground_truth(train_gt, 0.95, mode="random")
                # Generate the dataset
                from datautils import HSIDataset

                train_dataset = HSIDataset(
                    img, train_gt, window_size=hyperparams["patch_size"], overlap=TRAIN_OVERLAP
                )
                print("Sample in dataset: {}".format(len(train_dataset)))
                train_loader = data.DataLoader(
                    train_dataset,
                    batch_size=hyperparams["batch_size"],
                    # pin_memory=hyperparams['device'],
                    shuffle=True,
                    num_workers=N_JOBS,
                )
                # val_dataset = HyperX(img, val_gt, **hyperparams)
                val_dataset = HSIDataset(
                    img, test_gt, window_size=hyperparams["patch_size"], overlap=0.0
                )
                val_loader = data.DataLoader(
                    val_dataset,
                    # pin_memory=hyperparams['device'],
                    batch_size=hyperparams["batch_size"],
                    num_workers=N_JOBS,
                )

                # print(hyperparams)
                # print("Network :")
                # with torch.no_grad():
                #    for input, _ in train_loader:
                #        break
                #    summary(model.to(hyperparams["device"]), input.size()[1:])
                # We would like to use device=hyperparams['device'] altough we have
                # to wait for torchsummary to be fixed first.
                
        # Neural network
        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)

        if CHECKPOINT is not None:
            model.load_state_dict(torch.load(CHECKPOINT))

        try:
            train(
                model,
                optimizer,
                loss,
                train_loader,
                hyperparams["epoch"],
                exp_name=DATASET,
                scheduler=hyperparams["scheduler"],
                device=hyperparams["device"],
                supervision=hyperparams["supervision"],
                val_loader=val_loader,
                writer=writer,
            )
        except KeyboardInterrupt:
            # Allow the user to stop the training
            # TODO: move interruption inside training function?
            pass

        from utils import camel_to_snake
        save_model(
            model,
            camel_to_snake(str(model.__class__.__name__)),
            DATASET,
            epoch="last",
            metric=0,
        )
        if isinstance(img,list):
            probabilities = test(model, img[0][0].data, window_size=hyperparams["patch_size"],
                                 n_classes=len(LABEL_VALUES), overlap=TEST_OVERLAP)
        else:
            probabilities = test(model, img, window_size=hyperparams["patch_size"],
                                 n_classes=len(LABEL_VALUES), overlap=TEST_OVERLAP)
        prediction = np.argmax(probabilities, axis=-1)
        
        if isinstance(img, np.ndarray):
            from utils import pad_image
            window_size = PATCH_SIZE
            if isinstance(window_size, int):
                window_size = (window_size, window_size)
            padding = (window_size[0] // 2, window_size[1] // 2)
            from datautils import IGNORED_INDEX
            test_gt = pad_image(
                test_gt, padding=padding, mode="constant", constant=IGNORED_INDEX,
            ).astype("int64")
            run_results = metrics(prediction, test_gt, target_names=LABEL_VALUES)

            # mask = np.zeros(gt.shape, dtype="bool")
            # for l in IGNORED_LABELS:
            #     mask[gt == l] = True
            # prediction[mask] = 0

            results.append(run_results)
            show_results(run_results, writer)
        
        color_prediction = convert_to_color(prediction)
        display_predictions(
            color_prediction,
            writer,
            caption="Final prediction",
        )

    if N_RUNS > 1:
        show_results(results, writer, agregated=True)


if __name__ == "__main__":
    dataset_names = [
        v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
    ]

    # Argument parser for CLI interaction
    parser = argparse.ArgumentParser(
        description="Run deep learning experiments on" " various hyperspectral datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, choices=dataset_names, help="Dataset to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to train. Available:\n"
        "SVM (linear), "
        "SVM_grid (grid search on linear, poly and RBF kernels), "
        "baseline (fully connected NN), "
        "hu (1D CNN), "
        "hamida (3D CNN + 1D classifier), "
        "lee (3D FCN), "
        "chen (3D CNN), "
        "li (3D CNN), "
        "he (3D CNN), "
        "luo (3D CNN), "
        "sharma (2D CNN), "
        "boulch (1D semi-supervised CNN), "
        "liu (3D semi-supervised CNN), "
        "mou (1D RNN)",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Folder where to store the "
        "datasets (defaults to the current working directory).",
        default="./Datasets/",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=-1,
        help="Specify CUDA device (defaults to -1, which learns on CPU)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=0,
        help="Parallel threads for sklearn models and data loading/augmentation (0=blocking data loading, 1=asynchronous data loading, n>=2=parallel data loading)",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (default: 1)")
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        help="Weights to use for initialization, e.g. a checkpoint",
    )

    # Dataset options
    group_dataset = parser.add_argument_group("Dataset")
    group_dataset.add_argument(
        "--training_sample",
        type=float,
        default=10,
        help="Percentage of samples to use for training (default: 10%%)",
    )
    group_dataset.add_argument(
        "--sampling_mode",
        type=str,
        help="Sampling mode" " (random sampling or disjoint, default: random)",
        default="random",
    )
    group_dataset.add_argument(
    "--nblocks",
    type=int,
    help="Maximum number of blocks for blocks sampling_mode, e.g 8",
    default=None,
    )
    group_dataset.add_argument(
        "--train_set",
        type=str,
        default=None,
        help="Path to the train ground truth (optional, this "
        "supersedes the --sampling_mode option)",
    )
    group_dataset.add_argument(
        "--test_set",
        type=str,
        default=None,
        help="Path to the test set (optional, by default "
        "the test_set is the entire ground truth minus the training)",
    )
    # Training options
    group_train = parser.add_argument_group("Training")
    group_train.add_argument(
        "--epoch",
        type=int,
        help="Training epochs (optional, if" " absent will be set by the model)",
    )
    group_train.add_argument(
        "--patch_size",
        type=int,
        help="Size of the spatial neighbourhood (optional, if "
        "absent will be set by the model)",
    )
    group_train.add_argument(
        "--lr", type=float, help="Learning rate, set by the model if not specified."
    )
    group_train.add_argument(
        "--class_balancing",
        action="store_true",
        help="Inverse median frequency class balancing (default = False)",
    )
    group_train.add_argument(
        "--batch_size",
        type=int,
        help="Batch size (optional, if absent will be set by the model",
    )
    group_train.add_argument(
        "--train_overlap",
        type=float,
        default=0,
        help="Sliding window overlap stride during training (max 1, default = 0)"
    )
    group_train.add_argument(
        "--test_overlap",
        type=float,
        default=0,
        help="Sliding window overlap stride during inference (max 1, default = 0)",
    )
    # Data augmentation parameters
    group_da = parser.add_argument_group("Data augmentation")
    group_da.add_argument(
        "--flip_augmentation",
        action="store_true",
        help="Random flips (if patch_size > 1)",
    )
    group_da.add_argument(
        "--radiation_augmentation",
        action="store_true",
        help="Random radiation noise (illumination)",
    )
    group_da.add_argument(
        "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
    )

    parser.add_argument(
        "--with_exploration",
        action="store_true",
        help="See data exploration visualization",
    )
    parser.add_argument(
        "--download",
        type=str,
        default=None,
        nargs="+",
        choices=dataset_names,
        help="Download the specified datasets and quits.",
    )

    args = parser.parse_args()
    main(args)
