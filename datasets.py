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

DATASETS_CONFIG = {
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
         'Mandji_Z2': {
            'folder': 'Mandji/',
            'img': 'img_Mandji_ZONE2_hyper.hdr',
            'gt': 'Mandji_zone2_VT_synthese.hdr',
            'download': False
            },
         'Mandji_Z4': {
            'folder': 'Mandji/',
            'img': 'img_Mandji_ZONE4_hyper.hdr',
            'gt': 'Mandji_zone4_VT_synthese.hdr',
            'download': False
            },
         'DFC2018_HSI': {
            'img': '2018_IEEE_GRSS_DFC_HSI_TR.HDR',
            'gt': '2018_IEEE_GRSS_DFC_GT_TR.tif',
            'download': False
            }
    }

def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

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


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """

    
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', True):
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
    elif not os.path.isdir(folder):
       print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'PaviaC':
        # Load the image
        img = open_file(folder + 'Pavia.mat')['pavia']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'Pavia_gt.mat')['pavia_gt']

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]

        ignored_labels = [0]

    elif dataset_name == 'PaviaU':
        # Load the image
        img = open_file(folder + 'PaviaU.mat')['paviaU']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']

        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        ignored_labels = [0]

    elif dataset_name == 'IndianPines':
        # Load the image
        img = open_file(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]

        ignored_labels = [0]

    elif dataset_name == 'Botswana':
        # Load the image
        img = open_file(folder + 'Botswana.mat')['Botswana']

        rgb_bands = (75, 33, 15)

        gt = open_file(folder + 'Botswana_gt.mat')['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]

        ignored_labels = [0]

    elif dataset_name == 'KSC':
        # Load the image
        img = open_file(folder + 'KSC.mat')['KSC']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'KSC_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]
    elif dataset_name == 'Mandji_Z2':
        # Load the image
        img = open_file(folder + 'img_Mandji_ZONE2_hyper.hdr')

        rgb_bands = (60, 32, 10)

        # Extended SAM1 GT
        gt = open_file(folder + 'Mandji_zone2_VT_synthese.hdr')[:,:,1]
        gt = np.array(gt).squeeze().astype('uint8')
        label_values = ["non identifie", #0
                        "eau", #1
                        "laterite", #2
                        "sable gris", #3
                        "sable sec", #4
                        "sable humide", #5
                        "béton", #6
                        "cabane", #7
                        "végétation rase stressée", #8
                        "végétation rase + sol nu", #9
                        "végétation verte clairsemée", #10
                        "végétation verte dense", #11
                        "--", #12
                        "ancien marigot sud", #13
                        "marigot asséché", #14
                        "ancien marigot nord",#15
                       ]
        ignored_labels = [0]
    elif dataset_name == 'Mandji_Z4':
        # Load the image
        img = open_file(folder + 'img_Mandji_ZONE4_hyper.hdr')

        rgb_bands = (60, 32, 10)

        # Extended SAM1 GT
        gt = open_file(folder + 'Mandji_zone4_VT_synthese.hdr')[:,:,1]
        gt = np.array(gt).squeeze().astype('uint8')
        label_values = ["non identifie", #0
                        "eau", #1
                        "laterite", #2
                        "sable gris", #3
                        "sable sec", #4
                        "sable humide", #5
                        "béton", #6
                        "cabane", #7
                        "végétation rase stressée", #8
                        "végétation rase + sol nu", #9
                        "végétation verte clairsemée", #10
                        "végétation verte dense", #11
                        "--", #12
                        "ancien marigot sud", #13
                        "marigot asséché", #14
                        "hydrocarbure",#15
                         ]
        ignored_labels = [0]
    elif dataset_name == 'DFC2018_HSI':
        img = open_file(folder + '2018_IEEE_GRSS_DFC_HSI_TR.HDR')
        gt = open_file(folder + '2018_IEEE_GRSS_DFC_GT_TR.tif')
        gt = gt.astype('uint8')

        rgb_bands = (48, 32, 16)

        label_values = ["Unclassified",
                        "Healthy grass",
                        "Stressed grass",
                        "Artificial turf",
                        "Evergreen trees",
                        "Deciduous trees",
                        "Bare earth",
                        "Water",
                        "Residential buildings",
                        "Non-residential buildings",
                        "Roads",
                        "Sidewalks",
                        "Crosswalks",
                        "Major thoroughfares",
                        "Highways",
                        "Railways",
                        "Paved parking lots",
                        "Unpaved parking lots",
                        "Cars",
                        "Trains",
                        "Stadium seats"]
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
