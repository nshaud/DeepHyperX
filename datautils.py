IGNORED_INDEX = 255

import numpy as np
import torch

from utils import pad_image
from utils import sliding_window


def count_valid_pixels(arr, ignored=IGNORED_INDEX):
    return np.count_nonzero(arr != ignored)


class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, hsi_images, masks, window_size=None, overlap=0, step=None):
        super(HSIDataset, self).__init__()
        # Single image/mask pair are wrapped in a list
        if not isinstance(hsi_images, list):
            data = [data]
        if not isinstance(masks, list):
            masks = [masks]

        # Transform singular window size into a tuple
        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        # Padding = half the size of the window
        padding = (window_size[0] // 2, window_size[1] // 2)
        # Pad image and ground truth
        self.data = [
            pad_image(image, padding=padding).astype("float32") for image in hsi_images
        ]
        self.masks = [
            pad_image(
                mask, padding=padding, mode="constant", constant=IGNORED_INDEX
            ).astype("int64")
            for mask in masks
        ]
        self.window_size = window_size

        if step is None:
            # Overlap percentage defines how much two successive windows intersect
            # This directly gives the step size of the sliding window:
            #   0% overlap => step size = window size
            #   50% overlap => step size = 0.5 * window size
            #   90% overlap => step size = 0.9 * window size
            assert overlap >= 0 and overlap < 1
            step_h = int((1 - overlap) * window_size[0])
            step_w = int((1 - overlap) * window_size[1])
        elif isinstance(step, int):
            step_h, step_w = (step, step)
        else:
            step_h, step_w = step

        self.window_corners = []
        # Extract window corner indices
        for idx, (_, mask) in enumerate(zip(self.data, self.masks)):
            windows = list(
                sliding_window(
                    mask,
                    step=(step_h, step_w),
                    window_size=window_size,
                    with_data=True,
                )
            )
            # Skip windows that only contains ignored pixels
            self.window_corners += [
                (idx, (x, y))
                for window, x, y, w, h in windows
                if count_valid_pixels(window) > 0
            ]

    def __len__(self):
        # Dataset length is the number of windows
        return len(self.window_corners)

    def __getitem__(self, idx):
        w, h = self.window_size
        data_idx, (x, y) = self.window_corners[idx]
        # Extract window from image/ground truth
        data = self.data[data_idx][x : x + w, y : y + h].transpose((2, 0, 1))
        target = self.masks[data_idx][x : x + w, y : y + h]
        # TODO: data augmentation
        return torch.from_numpy(data), torch.from_numpy(target)


class HSITestDataset(HSIDataset):
    def __init__(self, hsi_image, window_size=None, overlap=0, step=None):
        masks = np.zeros(hsi_image.shape[:2], dtype="int64")
        super(HSITestDataset, self).__init__(
            hsi_image, masks, window_size=window_size, overlap=overlap, step=step
        )

    def __getitem__(self, idx):
        w, h = self.window_size
        x, y = self.window_corners[idx]
        data = self.data[x : x + w, y : y + h].transpose((2, 0, 1))
        # TODO: test time augmentation?
        coords = np.array([[x, x + w], [y, y + h]])
        return torch.from_numpy(data), torch.from_numpy(coords)


class HSICenterPixelDataset(HSIDataset):
    def __init__(self, hsi_image, masks, window_size=None):
        step = (1, 1)
        super(HSICenterPixelDataset, self).__init__(
            hsi_image, masks, window_size=window_size, step=step
        )

    def __getitem__(self, idx):
        data, target = super(HSICenterPixelDataset, self).__getitem__(idx)
        w, h = self.window_size
        target = target[..., w // 2, h // 2]
        return data, target


class HSICenterPixelTestDataset(HSICenterPixelDataset):
    def __init__(self, hsi_image, window_size=None):
        masks = np.zeros(hsi_image.shape[:2], dtype="int64")
        super(HSICenterPixelTestDataset, self).__init__(
            hsi_image, masks, window_size=window_size
        )

    def __getitem__(self, idx):
        w, h = self.window_size
        x, y = self.window_corners[idx]
        data = self.data[x : x + w, y : y + h].transpose((2, 0, 1))
        # TODO: test time augmentation?
        coords = np.array([[x, x + w], [y, y + h]])
        return torch.from_numpy(data), torch.from_numpy(coords)