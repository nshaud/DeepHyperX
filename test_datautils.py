import numpy as np
import pytest

from datautils import HSIDataset
from datautils import to_sklearn_datasets

@pytest.fixture
def image():
    return np.random.randn(100, 100, 50)

@pytest.fixture
def mask():
    return np.random.randint(0, 10, size=(100, 100), dtype="int64")

class TestHSIDataset:

    def test_patch_size_no_overlap(self, image, mask):
        window_size = (10, 10)
        patch_dims = (image.shape[-1], ) + window_size
        ds = HSIDataset(image, mask, window_size=window_size, overlap=0.0)
        equals = [img.shape == patch_dims and gt.shape == window_size for img, gt in ds]
        assert all(equals)

    def test_patch_size_overlap(self, image, mask):
        window_size = (10, 10)
        patch_dims = (image.shape[-1], ) + window_size
        ds = HSIDataset(image, mask, window_size=window_size, overlap=0.50)
        equals = [img.shape == patch_dims and gt.shape == window_size for img, gt in ds]
        assert all(equals)
    
    def test_number_of_patches(self, image, mask):
        window_size = (10, 10)
        ds = HSIDataset(image, mask, window_size=window_size, overlap=0.0)
        # 11x11 window grid
        assert len(ds) == 11*11
        ds = HSIDataset(image, mask, window_size=window_size, overlap=0.5)
        # 21x21 grid (with 50% overlap)
        assert len(ds) == 21*21

def test_sklearn_dataset_building(image, mask):
    # TODO: add a test with ignored pixels ?
    w, h, n_bands = image.shape
    X, y = to_sklearn_datasets(image, mask)
    assert len(X) == len(y)
    assert X.shape == (w*h, n_bands)

# TODO: add tests for HSITestDataset