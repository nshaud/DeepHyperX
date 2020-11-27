import numpy as np
import torch

from utils import pad_image
from utils import sliding_window

IGNORED_INDEX = 255

def count_valid_pixels(arr, ignored=IGNORED_INDEX):
    return np.count_nonzero(arr != ignored)

class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, hsi_image, ground_truth, window_size=None, overlap=0):
        super(HSIDataset, self).__init__()
        # Pad image and ground truth
        padding = (window_size[0] // 2, window_size[1] // 2)
        self.data = pad_image(hsi_image, padding=padding).astype("float32")
        self.ground_truth = pad_image(
            ground_truth,
            padding=padding,
            mode="constant",
            constant=IGNORED_INDEX,
        ).astype("int64")
        self.window_size = window_size
        self.ignored_mask = self.ground_truth == IGNORED_INDEX

        assert overlap >= 0 and overlap < 1
        step = int((1 - overlap) * min(window_size))

        windows = list(
            sliding_window(
                self.ground_truth, step=step, window_size=window_size, with_data=True
            )
        )
        # Skip windows that only contains ignored pixels
        self.window_corners = [
            (x, y)
            for window, x, y, w, h in windows
            if count_valid_pixels(window) > 0
        ]

    def __len__(self):
        return len(self.window_corners)

    def __getitem__(self, idx):
        w, h = self.window_size
        x, y = self.window_corners[idx]
        # Extract window from image/ground truth
        data = self.data[x : x + w, y : y + h]
        target = self.ground_truth[x : x + w, y : y + h]
        # TODO: data augmentation
        return torch.from_numpy(data), torch.from_numpy(target)
