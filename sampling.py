import numpy as np

from sklearn.model_selection import train_test_split

from datautils import IGNORED_INDEX


def middle_train_test_split(gt, train_size=0.5):
    train_gt = np.full_like(gt, IGNORED_INDEX)
    test_gt = np.full_like(gt, IGNORED_INDEX)
    for c in np.unique(gt):
        if c == IGNORED_INDEX:
            continue
        class_mask = gt == c
        ratios = np.zeros((gt.shape[0],))
        for line in range(gt.shape[0]):
            first_half_count = np.count_nonzero(class_mask[:line, :])
            second_half_count = np.count_nonzero(class_mask[line:, :])

            try:
                ratios[line] = first_half_count / (first_half_count + second_half_count)
            except ZeroDivisionError:
                ratios[line] = 1
        line = np.argmin(np.abs(ratios - train_size))
        print(f"Best found ratio = {ratios[line]:.2f} at line {line}")
        train_class_mask, test_class_mask = np.copy(class_mask), np.copy(class_mask)
        train_class_mask[line:, :] = False
        test_class_mask[:line, :] = False
        train_gt[train_class_mask] = c
        test_gt[test_class_mask] = c
    return train_gt, test_gt


def random_train_test_split(gt, train_size=0.5):
    valid_pixels = np.nonzero(gt != IGNORED_INDEX)
    train_x, test_x, train_y, test_y = train_test_split(
        *valid_pixels, train_size=train_size, stratify=gt[valid_pixels].ravel()
    )

    train_indices = (train_x, train_y)
    test_indices = (test_x, test_y)

    # Copy train/test pixels
    train_gt = np.full_like(gt, IGNORED_INDEX)
    test_gt = np.full_like(gt, IGNORED_INDEX)
    train_gt[train_indices] = gt[train_indices]
    test_gt[test_indices] = gt[test_indices]
    return train_gt, test_gt


def split_ground_truth(ground_truth, train_size, mode="random"):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        ground_truth: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    if mode == "random":
        train_gt, test_gt = random_train_test_split(ground_truth, train_size)
    elif mode == "disjoint":
        train_gt, test_gt = middle_train_test_split(ground_truth, train_size)
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt
