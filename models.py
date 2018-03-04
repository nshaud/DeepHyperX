# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
# utils
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
from utils import grouper, SpatialCrossMapLRN, CrossEntropy2d,\
                  sliding_window, count_sliding_window, get_display_type


def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    cuda = kwargs.setdefault('cuda', False)
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = kwargs.setdefault('weights', weights)
    if cuda:
        kwargs['weights'] = weights.cuda()

    if name == 'nn':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model = Baseline(n_bands, n_classes,
                         kwargs.setdefault('dropout', False))
        lr = kwargs.setdefault('learning_rate', 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 100)
    elif name == 'hamida':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = HamidaEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.0001)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'lee':
        kwargs.setdefault('epoch', 200)
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = False
        model = LeeEtAl(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        def criterion(x, y):
            return CrossEntropy2d(x, y, weight=kwargs['weights'])
    elif name == 'chen':
        patch_size = kwargs.setdefault('patch_size', 27)
        center_pixel = True
        model = ChenEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.003)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 400)
        kwargs.setdefault('batch_size', 100)
    elif name == 'li':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = LiEtAl(n_bands, n_classes, n_planes=4, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optim.SGD(model.parameters(),
                              lr=lr, momentum=0.9, weight_decay=0.0005)
        kwargs.setdefault('epoch', 200)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'hu':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model = HuEtAl(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 1000)
        kwargs.setdefault('batch_size', 100)
    elif name == 'he':
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        kwargs.setdefault('patch_size', 7)
        kwargs.setdefault('batch_size', 40)
        lr = kwargs.setdefault('learning_rate', 0.01)
        center_pixel = True
        model = HeEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        # For Adagrad, we need to load the model on GPU before creating the optimizer
        if cuda:
            model = model.cuda()
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'luo':
        # All  the  experiments  are  settled  by  the  learning  rate  of  0.1,
        # the  decay  term  of  0.09  and  batch  size  of  100.
        kwargs.setdefault('patch_size', 3)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('learning_rate', 0.1)
        center_pixel = True
        model = LuoEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.09)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'sharma':
        # We train our S-CNN from scratch using stochastic gradient descent with
        # momentum set to 0.9, weight decay of 0.0005, and with a batch size
        # of 60.  We initialize an equal learning rate for all trainable layers
        # to 0.05, which is manually decreased by a factor of 10 when the validation
        # error stopped decreasing. Prior to the termination the learning rate was
        # reduced two times at 15th and 25th epoch. [...]
        # We trained the network for 30 epochs
        kwargs.setdefault('batch_size', 60)
        kwargs.setdefault('epoch', 30)
        lr = kwargs.setdefault('lr', 0.05)
        center_pixel = True
        # We assume patch_size = 64
        kwargs.setdefault('patch_size', 64)
        model = SharmaEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        #scheduler = 15th and 25th / 10
    else:
        raise KeyError("{} model is unknown.".format(name))

    if cuda:
        model = model.cuda()
    kwargs.setdefault('data_augmentation', False)
    kwargs.setdefault('epoch', 100)
    kwargs.setdefault('batch_size', 100)
    kwargs['center_pixel'] = center_pixel
    return model, optimizer, criterion, kwargs


class Baseline(nn.Module):
    """
    Baseline network
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform(m.weight.data)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x, verbose=False):
        if verbose:
            print("Input size: {}".format(x.size()))
        x = F.relu(self.fc1(x))
        if verbose:
            print("Fc1 size: {}".format(x.size()))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if verbose:
            print("Fc2 size: {}".format(x.size()))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if verbose:
            print("Fc3 size: {}".format(x.size()))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        if verbose:
            print("Output size: {}".format(x.size()))
        return x

class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal(m.weight.data)
        

    def _get_final_flattened_size(self):
        x = torch.zeros(1, 1, self.input_channels)
        x = Variable(x, volatile=True)
        x = self.pool(self.conv(x))
        return x.numel() 

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
           kernel_size = input_channels // 10 + 1
        if pool_size is None:
           pool_size = kernel_size // 5 + 1
        self.input_channels = input_channels

        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x, verbose=False):
        x = x.unsqueeze(1)
        if verbose:
            print(x.size())
        x = self.conv(x)
        if verbose:
            print(x.size())
        x = F.relu(self.pool(x))
        if verbose:
            print(x.size())
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        if verbose:
            print(x.size())
        x = F.relu(self.fc2(x))
        if verbose:
            print(x.size())
        return x

class HamidaEtAl(nn.Module):
    """
    DEEP LEARNING APPROACH FOR REMOTE SENSING IMAGE ANALYSIS
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    Big Data from Space (BiDS'16)
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform(m.weight.data)

    def __init__(self, input_channels, n_classes, patch_size=5):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), padding=1)
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), padding=0)
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        x = torch.zeros((1, 1, self.input_channels,
                         self.patch_size, self.patch_size))
        x = Variable(x)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x)
        _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x, verbose=False):
        if verbose:
            print("Input size : {}".format(x.size()))
        x = F.relu(self.conv1(x))
        if verbose:
            print("Conv1 size : {}".format(x.size()))
        x = self.pool1(x)
        if verbose:
            print("Pool1 size : {}".format(x.size()))
        x = F.relu(self.conv2(x))
        if verbose:
            print("Conv2 size : {}".format(x.size()))
        x = self.pool2(x)
        if verbose:
            print("Pool2 size : {}".format(x.size()))
        x = F.relu(self.conv3(x))
        if verbose:
            print("Conv3 size : {}".format(x.size()))
        x = F.relu(self.conv4(x))
        if verbose:
            print("Conv4 size : {}".format(x.size()))
        x = x.view(-1, self.features_size)
        if verbose:
            print("Flatten size : {}".format(x.size()))
        x = self.fc(x)
        if verbose:
            print("Output size : {}".format(x.size()))
        return x


class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform(m.weight.data)

    def __init__(self, in_channels, n_classes):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_1x1 = nn.Conv3d(
            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(256, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = SpatialCrossMapLRN(256)
        self.lrn2 = SpatialCrossMapLRN(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x, verbose=False):
        # Inception module
        x_3x3 = self.conv_3x3(x)
        if verbose:
            print("Inception 3x3 size : {}".format(x_3x3.size()))
        x_1x1 = self.conv_1x1(x)
        if verbose:
            print("Inception 1x1 size : {}".format(x_1x1.size()))
        x = torch.cat([x_3x3, x_1x1], dim=1)
        if verbose:
            print("Concatenated size : {}".format(x.size()))
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)
        if verbose:
            print("Squeezed size : {}".format(x.size()))

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)
        if verbose:
            print("First residual block : {}".format(x.size()))

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)
        if verbose:
            print("Second residual block : {}".format(x.size()))

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        if verbose:
            print("Output size : {}".format(x.size()))
        return x


class ChenEtAl(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """
    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal(m.weight.data, std=0.001)

    def __init__(self, input_channels, n_classes, patch_size=27, n_planes=32):
        super(ChenEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        x = torch.zeros((1, 1, self.input_channels,
                         self.patch_size, self.patch_size))
        x = Variable(x)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x, verbose=False):
        if verbose:
            print("Input size : {}".format(x.size()))
        x = F.relu(self.conv1(x))
        if verbose:
            print("Conv1 size : {}".format(x.size()))
        x = self.pool1(x)
        if verbose:
            print("Conv1 size : {}".format(x.size()))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        if verbose:
            print("Conv2 size : {}".format(x.size()))
        x = self.pool2(x)
        if verbose:
            print("Pool2 size : {}".format(x.size()))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        if verbose:
            print("Conv3 size : {}".format(x.size()))
        x = self.dropout(x)
        x = x.view(-1, self.features_size)
        if verbose:
            print("Flattened size : {}".format(x.size()))
        x = self.fc(x)
        if verbose:
            print("Output size : {}".format(x.size()))
        return x


class LiEtAl(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform(m.weight.data)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,
                               (3, 3, 3), padding=(1, 0, 0))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        x = torch.zeros((1, 1, self.input_channels,
                         self.patch_size, self.patch_size))
        x = Variable(x)
        x = self.conv1(x)
        x = self.conv2(x)
        _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x, verbose=False):
        if verbose:
            print("Input size : {}".format(x.size()))
        x = F.relu(self.conv1(x))
        if verbose:
            print("Conv1 size : {}".format(x.size()))
        x = F.relu(self.conv2(x))
        if verbose:
            print("Conv2 size : {}".format(x.size()))
        x = x.view(-1, self.features_size)
        if verbose:
            print("Flattened size : {}".format(x.size()))
        x = self.fc(x)
        if verbose:
            print("Output size : {}".format(x.size()))
        return x

class HeEtAl(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight.data)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3,1,1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0,0,0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1,0,0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2,0,0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5,0,0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0,0,0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1,0,0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2,0,0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5,0,0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3,2,2), stride=(3,2,2))
        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        x = torch.zeros((1, 1, self.input_channels,
                         self.patch_size, self.patch_size))
        x = Variable(x)
        x = self.conv1(x)
        print(x.size())
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        print(x2_1.size(), x2_2.size(), x2_3.size(), x2_4.size())
        x = x2_1 + x2_2 + x2_3 + x2_4
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = self.conv4(x)
        _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x, verbose=False):
        if verbose:
            print("Input size : {}".format(x.size()))
        x = F.relu(self.conv1(x))
        if verbose:
            print("Conv1 size : {}".format(x.size()))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        if verbose:
            print("Conv2 size : {}".format(x.size()))
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        if verbose:
            print("Conv3 size : {}".format(x.size()))
        x = F.relu(self.conv4(x))
        if verbose:
            print("Conv4 size : {}".format(x.size()))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        if verbose:
            print("Flattened size : {}".format(x.size()))
        x = self.fc(x)
        if verbose:
            print("Output size : {}".format(x.size()))
        return x

class LuoEtAl(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight.data)

    def __init__(self, input_channels, n_classes, patch_size=3, n_planes=90):
        super(LuoEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like 
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully 
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Conv3d(1, 90, (24, 3, 3), padding=0, stride=(9,1,1))
        self.conv2 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        x = torch.zeros((1, 1, self.input_channels,
                         self.patch_size, self.patch_size))
        x = Variable(x)
        x = self.conv1(x)
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = self.conv2(x)
        _, c, w, h = x.size()
        return c * w * h

    def forward(self, x, verbose=False):
        if verbose:
            print("Input size : {}".format(x.size()))
        x = F.relu(self.conv1(x))
        if verbose:
            print("Conv1 size : {}".format(x.size()))
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        if verbose:
            print("Conv2 size : {}".format(x.size()))
        x = x.view(-1, self.features_size)
        if verbose:
            print("Flattened size : {}".format(x.size()))
        x = F.relu(self.fc1(x))
        if verbose:
            print("FC1 size : {}".format(x.size()))
        x = self.fc2(x)
        if verbose:
            print("Output size : {}".format(x.size()))
        return x


class SharmaEtAl(nn.Module):
    """
    HYPERSPECTRAL CNN FOR IMAGE CLASSIFICATION & BAND SELECTION, WITH APPLICATION
    TO FACE RECOGNITION
    Vivek Sharma, Ali Diba, Tinne Tuytelaars, Luc Van Gool
    Technical Report, KU Leuven/ETH Zürich
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal(m.weight.data)

    def __init__(self, input_channels, n_classes, patch_size=64):
        super(SharmaEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # An input image of size 263x263 pixels is fed to conv1
        # with 96 kernels of size 6x6x96 with a stride of 2 pixels
        self.conv1 = nn.Conv3d(1, 96, (input_channels, 6, 6), stride=(1,2,2))
        self.conv1_bn = nn.BatchNorm3d(96)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        #  256 kernels of size 3x3x256 with a stride of 2 pixels
        self.conv2 = nn.Conv3d(1, 256, (96, 3, 3), stride=(1,2,2))
        self.conv2_bn = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # 512 kernels of size 3x3x512 with a stride of 1 pixel
        self.conv3 = nn.Conv3d(1, 512, (256, 3, 3), stride=(1,1,1))
        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step

        self.features_size = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.fc1 = nn.Linear(self.features_size, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        x = torch.zeros((1, 1, self.input_channels,
                         self.patch_size, self.patch_size))
        x = Variable(x)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        print(x.size())
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t*c, w, h) 
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        print(x.size())
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t*c, w, h) 
        x = F.relu(self.conv3(x))
        print(x.size())
        _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x, verbose=False):
        if verbose:
            print("Input size : {}".format(x.size()))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        if verbose:
            print("Conv1 size : {}".format(x.size()))
        x = self.pool1(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t*c, w, h) 
        x = F.relu(self.conv2_bn(self.conv2(x)))
        if verbose:
            print("Conv2 size : {}".format(x.size()))
        x = self.pool2(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t*c, w, h) 
        x = F.relu(self.conv3(x))
        if verbose:
            print("Conv3 size : {}".format(x.size()))
        x = x.view(-1, self.features_size)
        if verbose:
            print("Flattened size : {}".format(x.size()))
        x = self.fc1(x)
        if verbose:
            print("FC1 size : {}".format(x.size()))
        x = self.dropout(x)
        x = self.fc2(x)
        if verbose:
            print("Output size : {}".format(x.size()))
        return x

def train(net, optimizer, criterion, data_loader, epoch,
          save_epoch=5, display_iter=50, cuda=True, display=None):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        cuda (optional): bool set to True to use CUDA/CUDNN
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    if cuda:
        net.cuda()

    model_name = str(datetime.datetime.now()) + "_{}.pth"

    # Set the network to training mode
    net.train()

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    win = None
    d_type = get_display_type(display)
    if d_type == 'visdom':
        display_iter = 1

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):

        # Save the weights
        if e % save_epoch == 0:
            if not os.path.isdir('./checkpoints/'):
                os.mkdir('./checkpoints/')
            torch.save(net.state_dict(), './checkpoints/' + model_name.format(e))

        # Run the training loop for one epoch
        for batch_idx, (data, target) in enumerate(data_loader):
            # Load the data into the GPU if required
            if cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data[0]
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])
            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                string = string.format(
                    e, epoch, batch_idx *
                    len(data), len(data) * len(data_loader),
                    100. * batch_idx / len(data_loader), mean_losses[iter_])

                if d_type == 'visdom' and win:
                    display.line(
                        X=np.arange(iter_ - display_iter, iter_),
                        Y=mean_losses[iter_ - display_iter:iter_],
                        win=win,
                        update='append'
                    )
                elif d_type == 'visdom':
                    win = display.line(
                        X=np.arange(0, iter_),
                        Y=mean_losses[:iter_],
                    )
                elif d_type == 'plt':
                    # Refresh the Jupyter cell output
                    clear_output()
                    print(string)
                    plt.plot(mean_losses[:iter_]) and plt.show()
                else:
                    tqdm.write(string)
            iter_ += 1
            del(data, target, loss)


def test(net, img, hyperparams, patch_size=3,
         center_pixel=True, batch_size=25, cuda=True):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, cuda = hyperparams['batch_size'], hyperparams['cuda']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': 1, 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        if patch_size == 1:
            data = [b[0][0, 0] for b in batch]
            data = np.copy(data)
            data = torch.from_numpy(data)
        else:
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose(0, 3, 1, 2)
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

        indices = [b[1:] for b in batch]
        data = Variable(data, volatile=True)
        if cuda:
            data = data.cuda()
            output = net(data).data.cpu()
        else:
            output = net(data).data

        if patch_size == 1 or center_pixel:
            output = output.numpy()
        else:
            output = np.transpose(output.numpy(), (0, 2, 3, 1))
        for (x, y, w, h), out in zip(indices, output):
            if center_pixel:
                probs[x + w // 2, y + h // 2] += out
            else:
                probs[x:x + w, y:y + h] += out
    return probs
