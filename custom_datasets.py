from astropy.io import fits
import numpy as np

CUSTOM_DATASETS_CONFIG = {
    "Taurus250_sample": {
        "img": "taurus_L1495_250_sample.fits",
        "gt": "taurus_L1495_250_sample_mask.fits",
        "download": False,
        "loader": lambda folder: taurus_sample_loader("./Taurus/"),
    }
}
    
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
    ignored_labels = [0]
    palette = {0:(255,255,255),
    1:(128,128,128),
    2:(255,0,0)}
    return img, gt, rgb_bands, ignored_labels, label_values, palette
