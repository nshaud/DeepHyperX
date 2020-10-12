from astropy.io import fits
import numpy as np

CUSTOM_DATASETS_CONFIG = {
    "Taurus250_train": {
        "img": "taurus_L1495_250_train.fits",
        "gt": "taurus_L1495_250_train_mask.fits",
        "download": False,
        "loader": lambda folder: astro_loader("/Users/robitaij/postdoc/Lhyrica/Taurus/"),
    },
    "Taurus250_test": {
        "img": "taurus_L1495_250_test.fits",
        "gt": "taurus_L1495_250_test_mask.fits",
        "download": False,
        "loader": lambda folder: astro_loader("/Users/robitaij/postdoc/Lhyrica/Taurus/"),
    }
}


def taurus_train_loader(folder):
    img = fits.open(folder + "taurus_L1495_250_train.fits")[0].data[:, :, :-2]
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
    img = fits.open(folder + "taurus_L1495_250_test.fits")[0].data[:, :, :-2]
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
