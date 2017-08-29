"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
# -*- coding: utf-8 -*-
from scipy.io import loadmat
import numpy as np
import torch
import torch.utils


def get_dataset(dataset_name, data_sources=None):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        data_sources: paths to the datasets
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    try:
        DATA_FOLDER = data_sources[dataset_name]
    except KeyError:
        raise Exception("{} dataset is unknown. Available datasets: {}".format(
            dataset_name, data_sources.keys()))

    if dataset_name == 'PaviaC':
        # Load the image
        img = loadmat(DATA_FOLDER + 'Pavia.mat')['pavia']
        img = np.asarray(img, dtype='float32')

        rgb_bands = (55, 41, 12)

        gt = loadmat(DATA_FOLDER + 'Pavia_gt.mat')['pavia_gt']

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]

        ignored_labels = [0]

    elif dataset_name == 'PaviaU':
        # Load the image
        img = loadmat(DATA_FOLDER + 'PaviaU.mat')['paviaU']
        img = np.asarray(img, dtype='float32')

        rgb_bands = (55, 41, 12)

        gt = loadmat(DATA_FOLDER + 'PaviaU_gt.mat')['paviaU_gt']

        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        ignored_labels = [0]

    elif dataset_name == 'IndianPines':
        # Load the image
        img = loadmat(DATA_FOLDER + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']
        img = np.asarray(img, dtype='float32')

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = loadmat(DATA_FOLDER + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]

        ignored_labels = [0]

    elif dataset_name == 'Botswana':
        # Load the image
        img = loadmat(DATA_FOLDER + 'Botswana.mat')['Botswana']
        img = np.asarray(img, dtype='float32')

        rgb_bands = (75, 33, 15)

        gt = loadmat(DATA_FOLDER + 'Botswana_gt.mat')['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]

        ignored_labels = [0]

    elif dataset_name == 'KSC':
        # Load the image
        img = loadmat(DATA_FOLDER + 'KSC.mat')['KSC']
        img = np.asarray(img, dtype='float32')

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = loadmat(DATA_FOLDER + 'KSC_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]
    return img, gt, label_values, ignored_labels, rgb_bands


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, patch_size=3, ignored_labels=None,
                 center_pixel=False, data_augmentation=False):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.ignored_labels = set(ignored_labels)
        self.data_augmentation = data_augmentation
        self.center_pixel = center_pixel
        mask = np.ones_like(gt)
        for l in ignored_labels:
            mask[gt == l] = 0
        positions = np.nonzero(mask)
        self.x_indices, self.y_indices = positions

    def __len__(self):
        return len(self.x_indices)

    def __getitem__(self, i):
        x, y = self.x_indices[i], self.y_indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x + self.patch_size, y + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.data_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            if np.random.random() > 0.5:
                data = np.fliplr(data)
            if np.random.random() > 0.5:
                data = np.flipud(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        return data, label
