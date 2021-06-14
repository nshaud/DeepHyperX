# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import numpy as np
import seaborn as sns
import os
import yaml

from tqdm import tqdm
from urllib.request import urlretrieve

from datautils import IGNORED_INDEX
from utils import open_file

DEFAULT_CONFIG_FILE = "./datasets.yml"

with open(DEFAULT_CONFIG_FILE, "r") as f:
    CONFIGURATION = yaml.safe_load(f)


def download_from_url(target, url):
    with TqdmUpTo(
        unit="B", unit_scale=True, miniters=1, desc="Downloading {}".format(target)
    ) as t:
        urlretrieve(url, filename=target, reporthook=t.update_to)


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


class RawDataset(object):
    @classmethod
    def fromconfig(cls, name, datadir="./Datasets", download=False):
        config = CONFIGURATION[name]

        # Download if required or if data directory does not exist
        folder = os.path.join(datadir, config["folder"])
        directory_exists = os.path.isdir(folder)
        if download or not directory_exists:
            # Create folder if needed
            if not directory_exists:
                os.makedirs(folder)
            # Download files (#TODO: deal with the case where no URL is available)
            for data in config["data"]:
                image = data["image"]
                print(data)
                download_from_url(os.path.join(folder, image["name"]), image["url"])
                if "mask" in data.keys():
                    mask = data["mask"]
                    download_from_url(os.path.join(folder, mask["name"]), mask["url"])

        images, masks = [], []
        train_ids, test_ids = [], []
        for idx, data in enumerate(config["data"]):
            images.append(data["image"]["name"])
            masks.append(data["mask"]["name"])
            # List test identifiers
            is_test = data.get("test", False)
            if is_test:
                test_ids.append(idx)
            else:
                train_ids.append(idx)

        kwargs = {
            "ignored_labels": config.get("ignored_labels", []),
            "labels": config["labels"],
            "rgb": tuple(
                map(int, config["rgb"].split(","))
            ),  # bands to use for RGB composite,
            "palette": config.get("palette", None),
        }

        # Split dataset if specified in configuration
        train_data = [images[idx] for idx in train_ids]
        train_masks = [masks[idx] for idx in train_ids]
        test_data = [images[idx] for idx in test_ids]
        test_masks = [masks[idx] for idx in test_ids]

        train_dataset = RawDataset(train_data, train_masks, folder, **kwargs)
        if len(test_data) > 0:
            test_dataset = RawDataset(test_data, test_masks, folder, **kwargs)
            return train_dataset, test_dataset
        else:
            return train_dataset

    def __init__(
        self,
        data,
        masks,
        folder,
        labels=None,
        ignored_labels=None,
        rgb=None,
        palette=None,
    ):
        self.folder = folder
        self.labels = labels
        self.ignored_labels = ignored_labels
        self.rgb = rgb

        # Load data in memory
        self.data = [self.load_data(filename) for filename in data]
        self.masks = [self.load_mask(filename) for filename in masks]

        self.bands = self.data[0].shape[-1]  # number of spectral wavelengths
        # Consistency check
        self.check_consistency()

        self.rgb_bands = rgb
        if palette is None:
            # Generate color palette using seaborn HLS
            ids = range(len(self.labels))
            self.palette = {
                idx: tuple(np.asarray(255 * np.array(color), dtype="uint8"))
                for idx, color in zip(ids, sns.color_palette("hls", len(ids)))
            }
        self.palette[IGNORED_INDEX] = (0, 0, 0)

    @property
    def datalen(self):
        return len(self.data)

    def load_data(self, filename):
        path = os.path.join(self.folder, filename)
        image = open_file(path)
        return image

    def load_mask(self, filename):
        path = os.path.join(self.folder, filename)
        mask = open_file(path)
        # Remove ignored classes from the ground truth
        for idx in self.ignored_labels:
            mask[mask == idx] = IGNORED_INDEX
        return mask

    # Principles:
    # - we don't check for NaNs
    # - we don't change data without user input

    def to_sklearn_datasets(self):
        all_pixels, all_labels = [], []
        for image, ground_truth in zip(self.data, self.masks):
            valid_pixels = ground_truth != IGNORED_INDEX
            all_pixels.append(image[valid_pixels])
            all_labels.append(ground_truth[valid_pixels].ravel())
        samples = np.concatenate(all_pixels)
        labels = np.concatenate(all_labels)
        return samples, labels

    def check_consistency(self):
        assert len(self.data) == len(self.masks)
        for image, mask in zip(self.data, self.masks):
            # Image and label mask have same dimensions
            assert image.shape[:2] == mask.shape[:2]
            # All images have the same number of bands
            assert image.shape[-1] == self.bands

    # def split(self):


ksc = RawDataset.fromconfig("PaviaC")
print(ksc.__dict__)

    # # Relabel the classes based on what has been ignored
    # from sklearn.preprocessing import LabelEncoder

    # mask = gt == IGNORED_INDEX
    # le = LabelEncoder()
    # gt[~mask] = le.fit_transform(gt[~mask])#.reshape(gt.shape)

    # # Fix the palette after relabeling
    # palette = {
    #     new_idx: palette[old_idx]
    #     for new_idx, old_idx in enumerate(le.classes_)
    #     if old_idx != IGNORED_INDEX
    # }
    # palette[IGNORED_INDEX] = (0, 0, 0)
    # # Fix the label values after relabeling
    # label_values = [label_values[c] for c in le.classes_ if c != IGNORED_INDEX]
    # # Normalization
    # img = np.asarray(img, dtype="float32")
    # print(img.shape)
    # # TODO: make this configurable
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # return img, gt, label_values, rgb_bands, palette
