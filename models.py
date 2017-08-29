# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch
import torch.optim as optim
from torch.autograd import Variable, Function
# utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import visdom
from utils import *
from tqdm import tqdm


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
    weights = kwargs.setdefault('weights', torch.ones(n_classes))
    if cuda:
        kwargs['weights'] = weights.cuda()

    if name == 'baseline':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model = Baseline(n_bands, n_classes,
                         kwargs.setdefault('dropout', False))
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 20)
        kwargs.setdefault('batch_size', 256)
    elif name == 'hamida':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = HamidaEtAl(n_bands, n_classes, patch_size=patch_size)
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    elif name == 'lee':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = False
        model = LeeEtAl(n_bands, n_classes)
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        criterion = lambda x,y: CrossEntropy2d(x,y, weight=weights)
    elif name == 'chen':
        patch_size = kwargs.setdefault('patch_size', 23)
        center_pixel = True
        model = ChenEtAl(n_bands, n_classes, n_planes=32)
        optimizer = optim.SGD(model.parameters(), lr=0.003)
        criterion = nn.CrossEntropyLoss(weight=weights)
        kwargs.setdefault('epoch', 200)
        kwargs.setdefault('batch_size', 100)
    elif name == 'li':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = LiEtAl(n_bands, n_classes, n_planes=2, patch_size=patch_size)
        optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0005)
        kwargs.setdefault('epoch', 100)
        criterion = nn.CrossEntropyLoss(weight=weights)

    if cuda:
        model = model.cuda()

    kwargs.setdefault('epoch', 100)
    kwargs.setdefault('batch_size', 100)
    kwargs['center_pixel'] = center_pixel
    return model, optimizer, criterion, kwargs

class Baseline(nn.Module):
    """
    Baseline network
    """
    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

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

class HamidaEtAl(nn.Module):
    """
    DEEP LEARNING APPROACH FOR REMOTE SENSING IMAGE ANALYSIS
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    Big Data from Space (BiDS'16)
    """
    def __init__(self, input_channels, n_classes, patch_size=5):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels

        if patch_size == 3:
            self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=(1,1,1), padding=1)
        else:
            self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=(1,1,1), padding=0)
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride equal
        # to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(20, 20, (3, 1, 1), stride=(2,1,1), padding=(1,0,0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(20, 35, (3, 3, 3), stride=(1,1,1), padding=(1,0,0))
        self.pool2 = nn.Conv3d(35, 35, (3, 1, 1), stride=(2,1,1), padding=(1, 0, 0))
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(35, 35, (3, 1, 1), stride=(1,1,1), padding=(1,0,0))
        self.conv4 = nn.Conv3d(35, 35, (2, 1, 1), stride=(2,1,1), padding=(1,0,0))

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

    def _get_final_flattened_size(self):
        x = torch.zeros((1, 1, self.input_channels, self.patch_size, self.patch_size))
        x = Variable(x)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x)
        _, t, c, w, h = x.size()
        return t*c*w*h

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
    def __init__(self):
        super(LeeEtAl, self).__init__(input_channels, n_classes)
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(1, 128, (input_channels, 3, 3), stride=(1,1,1), padding=(0,1,1))
        self.conv_1x1 = nn.Conv3d(1, 128, (input_channels, 1, 1), stride=(1,1,1), padding=0)

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(256, 128, (1,1))
        self.conv2 = nn.Conv2d(128, 128, (1,1))
        self.conv3 = nn.Conv2d(128, 128, (1,1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1,1))
        self.conv5 = nn.Conv2d(128, 128, (1,1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1,1))
        self.conv7 = nn.Conv2d(128, 128, (1,1))
        self.conv8 = nn.Conv2d(128, n_classes, (1,1))

        self.lrn1 = SpatialCrossMapLRN(256)
        self.lrn2 = SpatialCrossMapLRN(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

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
    def __init__(self, input_channels, n_classes, n_planes=32):
        super(ChenEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes

        self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.05)

    def _get_final_flattened_size(self):
        x = torch.zeros((1, 1, self.input_channels, self.patch_size, self.patch_size))
        x = Variable(x)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        _, t, c, w, h = x.size()
        return t*c*w*h

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
        x = x.view(-1, self.features_size)
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
    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1,0,0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2*n_planes, (3, 3, 3), padding=(1,0,0))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

    def _get_final_flattened_size(self):
        x = torch.zeros((1, 1, self.input_channels, self.patch_size, self.patch_size))
        x = Variable(x)
        x = self.conv1(x)
        x = self.conv2(x)
        _, t, c, w, h = x.size()
        return t*c*w*h

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

def train(net, optimizer, criterion, data_loader, epoch,
          display_iter=50, cuda=True, visdom=None):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        cuda (optional): bool set to True to use CUDA/CUDNN
        display_iter (optional): number of iterations before refreshing the display (False/None to switch off).
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    if cuda:
        net.cuda()

    # Set the network to training mode
    net.train()

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    win = None

    for e in tqdm(range(1, epoch + 1)):
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
            mean_losses[iter_] = np.mean(losses[max(0,iter_-100):iter_+1])
            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        e, epoch, batch_idx * len(data), len(data) * len(data_loader),
                        100. * batch_idx / len(data_loader), mean_losses[iter_])

                if visdom and win:
                    visdom.line(
                        X=np.arange(iter_ - display_iter, iter_),
                        Y=mean_losses[iter_ - display_iter:iter_],
                        win=win,
                        update='append'
                    )
                    visdom.text(string)
                elif visdom:
                    win = visdom.line(
                        X=np.arange(0, iter_),
                        Y=mean_losses[:iter_],
                    )
                    visdom.text(string)
                else:
                    # Refresh the Jupyter cell output
                    clear_output()
                    print(string)
                    plt.plot(mean_losses[:iter_]) and plt.show()
            iter_ += 1

def test(net, img, hyperparams, patch_size=3,
         center_pixel=True, batch_size=25, cuda=True):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size, center_pixel = hyperparams['patch_size'], hyperparams['center_pixel']
    batch_size, cuda = hyperparams['batch_size'], hyperparams['cuda']
    n_classes = hyperparams['n_classes']

    kwargs = { 'step': 1, 'window_size': (patch_size, patch_size) }
    probs = np.zeros(img.shape[:2] + (n_classes,))

    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                  total=(count_sliding_window(img, **kwargs) // batch_size)):
        if patch_size == 1:
            data = [b[0][0,0] for b in batch]
            data = np.copy(data)
            data = torch.from_numpy(data)
        else:
            data = [b[0] for b in batch]
            data = np.copy(data)
            data = data.transpose(0,3,1,2)
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

        indices = [b[1:] for b in batch]
        data = Variable(data, volatile=True)
        if cuda:
            data = data.cuda()
            output = net(data).data.cpu()
        else:
            output = model(data).data

        if patch_size == 1 or center_pixel:
            output = output.numpy()
        else:
            output = np.transpose(output.numpy(), (0, 2, 3, 1))
        for (x,y,w,h), out in zip(indices, output):
            if center_pixel:
                probs[x+w//2, y+h//2] += out
            else:
                probs[x:x+w, y:y+h] += out
    return probs
