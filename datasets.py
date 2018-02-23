"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
# -*- coding: utf-8 -*-
from scipy import io, misc
import spectral
import numpy as np
import torch
import torch.utils
import os
from tqdm import tqdm
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve


def loader(dataset):
    ext = os.path.splitext(dataset)
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder=None):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder: folder to store the datasets
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """

    datasets = {
        'PaviaC': {
            'img': 'http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
            'gt': 'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'
            },
        'PaviaU': {
            'img': 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
            'gt': 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'
            },
        'KSC': {
            'img': 'http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
            'gt': 'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'
            },
        'IndianPines': {
            'img': 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
            'gt': 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'
            },
        'Botswana': {
            'img': 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
            'gt': 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'
            },
         'Mandji': {
            'img': 'Mandji.mat',
            'gt': 'Mandji_gt.mat'
         }
    }
    
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    folder = target_folder + dataset_name + '/'
    # Download the dataset if is not present
    if os.path.isdir(folder):
        for url in datasets[dataset_name].values():
            filename = url.split('/')[-1]
    if not os.path.isdir(folder):
        os.mkdir(folder)
        for url in datasets[dataset_name].values():
            # download the files
            filename = url.split('/')[-1]
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc="Downloading {}".format(filename)) as t:
                urlretrieve(url, filename=folder + filename,
                            reporthook=t.update_to)

    if dataset_name == 'PaviaC':
        # Load the image
        img = io.loadmat(folder + 'Pavia.mat')['pavia']

        rgb_bands = (55, 41, 12)

        gt = io.loadmat(folder + 'Pavia_gt.mat')['pavia_gt']

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]

        ignored_labels = [0]

    elif dataset_name == 'PaviaU':
        # Load the image
        img = io.loadmat(folder + 'PaviaU.mat')['paviaU']

        rgb_bands = (55, 41, 12)

        gt = io.loadmat(folder + 'PaviaU_gt.mat')['paviaU_gt']

        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        ignored_labels = [0]

    elif dataset_name == 'IndianPines':
        # Load the image
        img = io.loadmat(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = io.loadmat(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]

        ignored_labels = [0]

    elif dataset_name == 'Botswana':
        # Load the image
        img = io.loadmat(folder + 'Botswana.mat')['Botswana']

        rgb_bands = (75, 33, 15)

        gt = io.loadmat(folder + 'Botswana_gt.mat')['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]

        ignored_labels = [0]

    elif dataset_name == 'KSC':
        # Load the image
        img = io.loadmat(folder + 'KSC.mat')['KSC']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = io.loadmat(folder + 'KSC_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]
    elif dataset_name == 'Mandji':
        # Load the image
        img = io.loadmat(folder + 'Mandji.mat')['mandji']

        rgb_bands = (60, 32, 10)

        gt = io.loadmat(folder + 'Mandji_gt.mat')['mandji_gt']
        gt = gt.astype('uint8')
        label_values = ["non identifie", #0
                        "eau", #1
                        "laterite", #2
                        "sable gris", #3
                        "béton", #4
                        "ancien marigot sud", #5
                        "ancien marigot nord",#6
                        "", #7
                        "végétation rase stressée", #8
                        "pipeline/merlon", #9
                        "vegetation dense verte", #10
                        "végétation autre"] #11

        ignored_labels = [0]

    # Normalization
    img = np.asarray(img, dtype='float32')
    img /= np.max(img)

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
        offset = patch_size // 2
        self.data = np.pad(data,
                           ((offset, offset), (offset, offset), (0, 0)),
                           'constant')
        self.label = np.pad(gt,
                            ((offset, offset), (offset, offset)),
                            'constant')
        self.patch_size = patch_size
        self.ignored_labels = set(ignored_labels)
        self.data_augmentation = data_augmentation
        self.center_pixel = center_pixel
        mask = np.ones_like(gt)
        for l in ignored_labels:
            mask[gt == l] = 0
        positions = np.nonzero(mask)
        x_pos = positions[0] + offset
        y_pos = positions[1] + offset
        self.indices = [idx for idx in zip(x_pos, y_pos)]
        np.random.shuffle(self.indices)

    @classmethod
    def augment_data(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5

        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x + self.patch_size // 2 + 1, y + self.patch_size // 2 + 1

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.data_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.augment_data(data, label)

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
