import numpy as np

import random

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from skimage.measure import label, regionprops

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

def blockmask(xpc, ypc, sz, arr):
    maskblock = np.zeros(arr)
    for elem in range(xpc.size):   
        maskblock[int(ypc[elem] - sz):int(ypc[elem] + sz),
				  int(xpc[elem] - sz):int(xpc[elem] + sz)] += 1
    return maskblock

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def genblocks(mask, nblocksmax=10):
	
    #########
    #Transform mask
    #########

    #mask[mask == 2] = 0

    #########
    #Read core positions
    #########

    labels, num = label(mask == np.max(mask),return_num=True)
    print("num=",num)
    xp = []
    yp = []
    for region in regionprops(labels):
        xp.append(region.centroid[1])
        yp.append(region.centroid[0])
    xp = np.asarray(xp)
    yp = np.asarray(yp)
    print("xp, yp sizes:",xp.size,yp.size)

    #########
    #Clustering cores with Kmeans
    #########

    Xpos = np.zeros((xp.size,2))
    Xpos[:,0] = xp
    Xpos[:,1] = yp
    kmeans = KMeans(n_clusters=nblocksmax)
    k_pred = kmeans.fit_predict(Xpos)

    #########
    #Estimate the size of the clusters and keep the smallest
    #########
    Sclust = np.zeros(nblocksmax)

    for nk in range(nblocksmax):
        pos = np.where(k_pred == nk)
        Sclust[nk] = np.max(np.sqrt(np.abs(xp[pos] - kmeans.cluster_centers_[nk,0])**2. + 
                                    np.abs(yp[pos] - kmeans.cluster_centers_[nk,1])**2.)) * 2.
    minSclust = np.min(Sclust)

    sz = int(minSclust/2)
    print("sz=",sz)

    #########
    #Sort core blocks
    #########
    order = np.argsort(kmeans.cluster_centers_[:,0])
    xpc = kmeans.cluster_centers_[order,0]
    ypc = kmeans.cluster_centers_[order,1]

    maskblock = blockmask(xpc, ypc, sz, mask.shape)

    #########
    #Update block positions
    #########
    while (np.where(maskblock == 2)[0].size !=0):
        corefill = np.zeros(xpc.size)
        flag = []
        for elem in range(xpc.size):
            #blocks outside image frame
            if ((ypc[elem] - sz) < 0):
                ypc[elem] += np.abs(ypc[elem] - sz)
            if ((ypc[elem] + sz) > mask.shape[0]):
                ypc[elem] = int(mask.shape[0]-sz)
            if ((xpc[elem] - sz) < 0):
                xpc[elem] += np.abs(xpc[elem] - sz)
            if ((xpc[elem] + sz) > mask.shape[1]):
                xpc[elem] = int(mask.shape[1]-sz)
            #remove overlaps
            canv = maskblock[int(ypc[elem] - sz):int(ypc[elem] + sz),
                             int(xpc[elem] - sz):int(xpc[elem] + sz)]
            ovl = np.where(canv == 2)
            if (ovl[0].size > 0):
                if (ovl[0].size < (sz*1.5)**2):
                    xu = np.unique(ovl[1])
                    yu = np.unique(ovl[0])
                    splitxu = consecutive(xu)
                    splityu = consecutive(yu)
                    #for lapse in range(len(splitxu)):
                    lapse = 0
                    if (splitxu[lapse].size < splityu[lapse].size) and (splitxu[lapse][0] < sz):
                        xpc[elem] += (np.max(splitxu[lapse])+1)
                    if (splityu[lapse].size < splitxu[lapse].size) and (splityu[lapse][0] < sz):
                        ypc[elem] += (np.max(splityu[lapse])+1)
                    maskblock = blockmask(xpc, ypc, sz, mask.shape)
                    nz = np.where(mask[int(ypc[elem] - sz):int(ypc[elem] + sz)
                                       ,int(xpc[elem] - sz):int(xpc[elem] + sz)] == np.max(mask))[0].size
                    corefill[elem] = float(nz) / (float(sz)*2.)**2
                    #canv = maskblock[int(ypc[elem] - sz):int(ypc[elem] + sz),int(xpc[elem] - sz):int(xpc[elem] + sz)]
                    #ovl = np.where(canv == 2)
                    #if (ovl[0].size > (sz*1.5)**2):
                    #    flag.append(elem)
                else:
                    flag.append(elem)
                    nz = np.where(mask[int(ypc[elem] - sz):int(ypc[elem] + sz),
                                       int(xpc[elem] - sz):int(xpc[elem] + sz)] == np.max(mask))[0].size
                    corefill[elem] = float(nz) / (float(sz)*2.)**2
            else:
                nz = np.where(mask[int(ypc[elem] - sz):int(ypc[elem] + sz),
                                   int(xpc[elem] - sz):int(xpc[elem] + sz)] == np.max(mask))[0].size
                corefill[elem] = float(nz) / (float(sz)*2.)**2

        #Filtering low core fill factor        
        filt = np.where((corefill-np.mean(corefill))/np.std(corefill) < -1.4)[0]

        if filt.size > 0:
            for elem in filt:
                flag.append(elem)

        flag = np.unique(flag).tolist()
        xpc = np.delete(xpc,flag)
        ypc = np.delete(ypc,flag)

        maskblock = blockmask(xpc, ypc, sz, mask.shape)

    #########
    #Background blocks
    #########

    xBclist = []
    yBclist = []

    xblock = np.int(mask.shape[1] / (sz*2))
    yblock = np.int(mask.shape[0] / (sz*2))

    for yl in range(yblock):
        for xl in range(xblock):
            BGblock = maskblock[yl*sz*2:yl*sz*2+sz*2,xl*sz*2:xl*sz*2+sz*2]
            if (np.where(BGblock == 1)[0].size == 0):
                xBclist.append(xl*sz*2+sz)
                yBclist.append(yl*sz*2+sz)

    xBc = np.array(xBclist)
    yBc = np.array(yBclist)

    return xpc, ypc, xBc, yBc, sz


def split_ground_truth(ground_truth, train_size, mode="random", **kwargs):
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
    elif mode == 'blocks':
        train_gt = np.full_like(ground_truth, IGNORED_INDEX)
        test_gt = np.full_like(ground_truth, IGNORED_INDEX)
        
        xpc, ypc, xBc, yBc, sz = genblocks(ground_truth, nblocksmax=kwargs.get('nblocks'))

        ##Train sample
        #Core blocks
        trainlist = range(xpc.size)
        trainsamp = random.sample(trainlist, k=round(xpc.size * train_size))
        train_mask = blockmask(xpc[trainsamp], ypc[trainsamp], sz, ground_truth.shape)
        idx = np.where(train_mask == 1)
        train_gt[idx] = ground_truth[idx]
        #Background blocks
        trainBlist = range(xBc.size)
        trainBsamp = random.sample(trainBlist, k=int(xBc.size * train_size))
        trainB_mask = blockmask(xBc[trainBsamp], yBc[trainBsamp], sz, ground_truth.shape)
        idxB = np.where(trainB_mask == 1)
        train_gt[idxB] = ground_truth[idxB]

        ##Test sample
        allblocks = blockmask(np.concatenate((xpc,xBc)), np.concatenate((ypc,yBc)), sz, ground_truth.shape)
        test_mask = allblocks * np.abs(train_mask - 1) * np.abs(trainB_mask - 1)
        idxall = np.where(test_mask == 1)
        test_gt[idxall] = ground_truth[idxall]
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt
