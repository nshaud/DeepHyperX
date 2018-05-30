from utils import open_file
import numpy as np

CUSTOM_DATASETS_CONFIG = {
         'Mandji_Z2': {
            'folder': 'Mandji/',
            'img': 'img_Mandji_ZONE2_hyper.hdr',
            'gt': 'Mandji_zone2_VT_synthese.hdr',
            'download': False,
            'loader': lambda folder: mandji_loader(folder, zone=2)
            },
         'Mandji_Z4': {
            'folder': 'Mandji/',
            'img': 'img_Mandji_ZONE4_hyper.hdr',
            'gt': 'Mandji_zone4_VT_synthese.hdr',
            'download': False,
            'loader': lambda folder: mandji_loader(folder, zone=4)
            },
         'DFC2018_HSI': {
            'img': '2018_IEEE_GRSS_DFC_HSI_TR.HDR',
            'gt': '2018_IEEE_GRSS_DFC_GT_TR.tif',
            'download': False,
            'loader': lambda folder: dfc2018_loader(folder)
            }
    }

def mandji_loader(folder, zone=2):
    if zone == 2:
        # Load the image
        img = open_file(folder + 'img_Mandji_ZONE2_hyper.hdr')
        img[np.isnan(img)] = 0.

        rgb_bands = (60, 32, 10)

        # Extended SAM1 GT
        gt = open_file(folder + 'Mandji_zone2_VT_synthese.hdr')[:,:,1]
        gt = np.array(gt).squeeze().astype('uint8')
        label_values = ["non identifie", #0
                        "eau", #1
                        "laterite", #2
                        "sable gris", #3
                        "ancien marigot (nord)", #4
                        "ancien marigot (sud)", #5
                        "béton", #6
                        "--", #7
                        "végétation rase stressée", #8
                        "végétation rase + sol nu", #9
                        "végétation verte clairsemée", #10
                        "végétation verte dense", #11
                       ]
        ignored_labels = [0]

        palette = {0: (0,0,0),
                   1: (0, 112, 192),
                   2: (247, 150, 70),
                   3: (166, 166, 166),
                   4: (112, 48, 160),
                   5: (254, 0, 236),
                   6: (255, 0, 0),
                   8: (255, 255, 0),
                   9: (153, 51, 0),
                   10: (0, 176, 80),
                   11: (0, 255, 0) 
                  }
        
    elif zone == 4:
        palette = {0: (0,0,0),
                   1: (0, 112, 192),
                   2: (247, 150, 70),
                   3: (166, 166, 166),
                   4: (112, 48, 160),
                   5: (254, 0, 236),
                   6: (255, 0, 0),
                   8: (255, 255, 0),
                   9: (153, 51, 0),
                   10: (0, 176, 80),
                   11: (0, 255, 0),
                   15: (255, 0, 0)
                  }
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

    return img, gt, rgb_bands, ignored_labels, label_values, palette

def dfc2018_loader(folder):
        img = open_file(folder + '2018_IEEE_GRSS_DFC_HSI_TR.HDR')[:,:,:-2]
        gt = open_file(folder + '2018_IEEE_GRSS_DFC_GT_TR.tif')
        gt = gt.astype('uint8')

        rgb_bands = (47, 31, 15)

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
        return img, gt, rgb_bands, ignored_labels, label_values, palette
