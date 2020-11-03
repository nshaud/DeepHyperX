from astropy.io import fits
import numpy as np

CUSTOM_DATASETS_CONFIG = {
    "Taurus250_train": {
        "img": "taurus_L1495_250_train.fits",
        "gt": "taurus_L1495_250_train_mask.fits",
        "download": False,
        "loader": lambda folder: taurus_train_loader("/Users/robitaij/postdoc/Lhyrica/Taurus/"),
    },
    "Taurus250_test": {
        "img": "taurus_L1495_250_test.fits",
        "gt": "taurus_L1495_250_test_mask.fits",
        "download": False,
        "loader": lambda folder: taurus_test_loader("/Users/robitaij/postdoc/Lhyrica/Taurus/"),
    },
    "Taurus250_sample": {
        "img": "taurus_L1495_250_sample.fits",
        "gt": "taurus_L1495_250_sample_mask.fits",
        "download": False,
        "loader": lambda folder: taurus_sample_loader("/Users/robitaij/postdoc/Lhyrica/Taurus/"),
    }
}


def taurus_train_loader(folder):
    img = fits.open(folder + "taurus_L1495_250_train.fits")[0].data
    gt = fits.open(folder + "taurus_L1495_250_train_mask.fits")[0].data
    gt = gt.astype("uint8")

    rgb_bands = (47, 31, 15)

    label_values = [
        "Unclassified",
        "Starless",
        "prestellar",
        "protostellar",
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette
    
def taurus_test_loader(folder):
    img = fits.open(folder + "taurus_L1495_250_test.fits")[0].data
    gt = fits.open(folder + "taurus_L1495_250_test_mask.fits")[0].data
    gt = gt.astype("uint8")

    rgb_bands = (47, 31, 15)

    label_values = [
        "Unclassified",
        "Starless",
        "prestellar",
        "protostellar",
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette
    
def taurus_sample_loader_old(folder):
    img = fits.open(folder + "taurus_L1495_250_sample.fits")[0].data
    gt = fits.open(folder + "taurus_L1495_250_sample_mask.fits")[0].data
    gt = gt.astype("uint8")

    rgb_bands = (0,0,0)

    label_values = [
    	"Unclassified",
        "background",
        "Starless",
        "prestellar",
        "protostellar",
    ]
    ignored_labels = []
    palette = {0:(255,255,255),
    1:(128,128,128),
    2:(255,0,0),
    3:(0,255,0),
    4:(0,0,255)}
    return img, gt, rgb_bands, ignored_labels, label_values, palette
    
def taurus_sample_loader(folder):
    img = fits.open(folder + "taurus_L1495_250_sample.fits")[0].data
    gt = fits.open(folder + "taurus_L1495_250_sample_mask.fits")[0].data
    gt = gt.astype("uint8")

    rgb_bands = (0,0,0)

    label_values = [
    	"Unclassified",
        "background",
        "cores",
    ]
    ignored_labels = []
    palette = {0:(255,255,255),
    1:(128,128,128),
    2:(255,0,0)}
    return img, gt, rgb_bands, ignored_labels, label_values, palette
