from astropy.io import fits
import numpy as np
from utils import open_file

CUSTOM_DATASETS_CONFIG = {
    "Taurus250_sample": {
        "img": "taurus_L1495_250_sample_log.fits",
        "gt": "taurus_L1495_250_sample_2nd_cat_mask.fits",
        "download": False,
        "loader": lambda folder: taurus_sample_loader("./Taurus/"),
    },
    "Taurus_cdens":{
        "img": "hi.surface.density.r18p2b_cent_norm_dhx.fits",
        "gt": "hi.surface.density.r18p2_mask2.fits",
        "download": False,
        "loader": lambda folder:taurus_cdens_loader("./Taurus/"),
    },
    "NGC2264_mwl": {
        "img": "ngc2264_mwl_centred_norm_log_dhx.fits",
        "gt": "ngc2264_mwl_mask.fits",
        "download": False,
        "loader": lambda folder: NGC2264_mwl_loader("./NGC2264/"),
    },
    "Multifractal_simu": {
        "img": "simu2048_gauss_norm_cent.fits",
        "gt": "simu2048_gauss_mask.fits",
        "download": False,
        "loader": lambda folder: simu_loader("./simu/"),
    },
	"Multi_image": {
        "img": "Multiple images",
        "gt": "Multiple masks",
        "download": False,
        "loader": lambda folder: multi_loader("/home/jrobitaille/lhyrica/simu/"),
    },
	"Benchmark": {
        "img": "sky_c175.hi.surface.density.r11p0_norm_dhx.fits",
        "gt": "sky_c250_mask.fits",
        "download": False,
        "loader": lambda folder: benchmark_loader("./benchmark/"),
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
    img = fits.open(folder + "hi.surface.density.r18p2b_cent_norm_dhx.fits")[0].data
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
    img = fits.open(folder + "ngc2264_mwl_centred_norm_log_dhx.fits")[0].data
    gt = fits.open(folder + "ngc2264_mwl_mask.fits")[0].data
    gt = gt.astype("uint8")

    rgb_bands = (0,1,2)

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
    img = fits.open(folder + "simu2048_gauss_norm_cent.fits")[0].data
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
    imgpath = []
    gtpath = []
    for ii in range(100):
        imgpath.append(folder+"var_dist_list/simu2048list_gauss{}_norm.fits".format(ii))
        gtpath.append(folder+"var_dist_list/simu2048list_gauss{}_mask.fits".format(ii))
        imgpath.append(folder+"no_cores/simu2048list_gauss{}_norm.fits".format(ii))
        gtpath.append(folder+"no_cores/simu2048list_gauss{}_mask.fits".format(ii))
    
    # Load all data
    imglist = []
    for ii in imgpath:
        imglist.append(fits.open(ii,memmap=True))
    gtlist = []
    for ii in gtpath:
        gtlist.append(fits.open(ii,memmap=True))

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
    return imglist, gtlist, rgb_bands, ignored_labels, label_values, palette

def benchmark_loader(folder):
    img = fits.open(folder + "sky_c175.hi.surface.density.r11p0_norm_dhx.fits")[0].data
    gt = fits.open(folder + "sky_c250_mask.fits")[0].data
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