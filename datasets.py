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
    with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc="Downloading {}".format(target)) as t:
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
    def __init__(self, name, datadir="./Datasets", download=False):
        config = CONFIGURATION[name]

        # Download if required or if data directory does not exist
        folder = os.path.join(datadir, config["folder"])
        directory_exists = os.path.isdir(folder)
        if download or not directory_exists:
            # Create folder if need
            if not directory_exists:
                os.makedirs(folder)
            # Download files (#TODO: deal with the case where no URL is available)
            for image in config["data"]:
                download_from_url(os.path.join(folder, image["name"]), image["url"])
            for mask in config["masks"]:
                download_from_url(os.path.join(folder, mask["name"]), mask["url"])
        
        self.folder = folder
        self.ignored_labels = config.get("ignored_labels", [])
        self.labels = config["labels"]
        self.data = [self.load_data(image["name"]) for image in config["data"]]
        self.masks = [self.load_mask(mask["name"]) for mask in config["masks"]]
        self.rgb_bands = tuple(map(int, config["rgb"].split(",")))

        self.palette = config.get("palette", None) # TODO
        if self.palette is None:
            # Generate color palette using seaborn HLS
            ids = range(len(self.labels))
            self.palette = {
                idx: tuple(np.asarray(255 * np.array(color), dtype="uint8"))
                for idx, color in zip(ids, sns.color_palette("hls", len(ids)))
            }
        self.palette[IGNORED_INDEX] = (0, 0, 0)

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
