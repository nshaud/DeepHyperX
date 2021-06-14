import numpy as np
import pytest

from datautils import count_valid_pixels
from datautils import IGNORED_INDEX

from sampling import middle_train_test_split
from sampling import random_train_test_split

@pytest.fixture
def ground_truth():
    gt = np.zeros((160, 80), dtype='uint8')
    gt[:20, :30] = 1
    gt[140:160,50:65] = 1
    gt[30:80, 20:45] = 2
    gt[100:140, 60:] = 3
    gt[45:60, 30:] = 4
    gt[100:120, 15:25] = 5
    gt[0:5,:] = IGNORED_INDEX
    return gt

def test_middle_split(ground_truth):
    train, test = middle_train_test_split(ground_truth, train_size=0.3)
    m1 = train != IGNORED_INDEX
    m2 = test != IGNORED_INDEX
    m = ground_truth != IGNORED_INDEX
    # Union of the two masks encompass the whole ground truth
    assert(np.all(np.bitwise_or(m1, m2) == m))
    # Intersection of the two masks is empty
    assert(not np.any(np.bitwise_and(m1, m2)))

def test_random_split(ground_truth):
    train, test = random_train_test_split(ground_truth, train_size=0.3)
    m1 = train != IGNORED_INDEX
    m2 = test != IGNORED_INDEX
    m = ground_truth != IGNORED_INDEX
    # Union of the two masks encompass the whole ground truth
    assert(np.all(np.bitwise_or(m1, m2) == m))
    # Intersection of the two masks is empty
    assert(not np.any(np.bitwise_and(m1, m2)))

def test_ratio(ground_truth):
    ratio = 0.3
    train, test = random_train_test_split(ground_truth, train_size=ratio)
    for c in np.unique(train):
        if c == IGNORED_INDEX:
            continue
        r = np.count_nonzero(train == c) / np.count_nonzero(ground_truth == c)
        # Tolerance 1%
        assert(r == pytest.approx(ratio, 0.01))
        r = np.count_nonzero(test == c) / np.count_nonzero(ground_truth == c)
        assert(r == pytest.approx(1 - ratio, 0.01))
    train, test = middle_train_test_split(ground_truth, train_size=ratio)
    for c in np.unique(train):
        if c == IGNORED_INDEX:
            continue
        r = np.count_nonzero(train == c) / np.count_nonzero(ground_truth == c)
        # Tolerance 20% for this one
        assert(r == pytest.approx(ratio, 0.2))
        r = np.count_nonzero(test == c) / np.count_nonzero(ground_truth == c)
        assert(r == pytest.approx(1 - ratio, 0.2))