from astropy.io import fits
import numpy as np

CUSTOM_DATASETS_CONFIG = {
    "Taurus250_sample": {
        "img": "taurus_L1495_250_sample_log.fits",
        "gt": "taurus_L1495_250_sample_2nd_cat_mask.fits",
        "download": False,
        "loader": lambda folder: taurus_sample_loader("./Taurus/"),
    },
    "Taurus_cdens":{
        "img": "hi.surface.density.r18p2_log_norm.fits",
        "gt": "hi.surface.density.r18p2_mask2.fits",
        "download": False,
        "loader": lambda folder:taurus_cdens_loader("./Taurus/"),
    },
    "NGC2264_mwl": {
        "img": "ngc2264_mwl_dhx.fits",
        "gt": "ngc2264_mwl_mask.fits",
        "download": False,
        "loader": lambda folder: NGC2264_mwl_loader("./NGC2264/"),
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
        "loader": lambda folder: multi_loader("./simu/"),
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
    img = fits.open(folder + "taurus_L1495_250_sample_log.fits")[0].data
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

def taurus_cdens_loader(folder):
    img = fits.open(folder + "hi.surface.density.r18p2_log_norm.fits")[0].data
    gt = fits.open(folder + "hi.surface.density.r18p2_mask2.fits")[0].data
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

def NGC2264_mwl_loader(folder):
    img = fits.open(folder + "ngc2264_mwl_dhx.fits")[0].data
    gt = fits.open(folder + "ngc2264_mwl_mask.fits")[0].data
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
