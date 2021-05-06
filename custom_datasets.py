from astropy.io import fits
import numpy as np

CUSTOM_DATASETS_CONFIG = {
    "Taurus250_sample": {
        "img": "taurus_L1495_250_sample.fits",
        "gt": "taurus_L1495_250_sample_2nd_cat_mask.fits",
        "download": False,
        "loader": lambda folder: taurus_sample_loader("./Taurus/"),
    },
    "Multifractal_simu": {
        "img": "simu2048_gauss.fits",
        "gt": "simu2048_gauss_mask.fits",
        "download": False,
        "loader": lambda folder: simu_loader("./simu/"),
    },
	"Multi_image": {
        "img": "Multiple images",
        "gt": "Multiple masks",
        "download": False,
        "loader": lambda folder: multi_loader("/Users/robitaij/postdoc/Lhyrica/simu/"),
    }
}
    
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
    gt = fits.open(folder + "taurus_L1495_250_sample_2nd_cat_mask.fits")[0].data
    gt = gt.astype("uint8")

    rgb_bands = (0,0,0)

    label_values = [
    	"Unclassified",
        "background",
        "cores",
    ]
    ignored_labels = [0]
    palette = {0:(255,255,255),
    1:(128,128,128),
    2:(255,0,0)}
    return img, gt, rgb_bands, ignored_labels, label_values, palette

def simu_loader(folder):
    img = fits.open(folder + "simu2048_gauss.fits")[0].data
    gt = fits.open(folder + "simu2048_gauss_mask.fits")[0].data
    gt = gt.astype("uint8")

    rgb_bands = (0,0,0)

    label_values = [
    	"Unclassified",
        "background",
        "cores",
    ]
    ignored_labels = [0]
    palette = {0:(255,255,255),
    1:(128,128,128),
    2:(255,0,0)}
    return img, gt, rgb_bands, ignored_labels, label_values, palette

def multi_loader(folder):
    img = [folder+"simu2048_gauss.fits",folder+"simu2048_gauss2.fits"]
    gt = [folder+"simu2048_gauss_mask.fits",folder+"simu2048_gauss2_mask.fits"]

    rgb_bands = (0,0,0)

    label_values = [
    	"Unclassified",
        "background",
        "cores",
    ]
    ignored_labels = [0]
    palette = {0:(255,255,255),
    1:(128,128,128),
    2:(255,0,0)}
    return img, gt, rgb_bands, ignored_labels, label_values, palette