# DeepHyperX

A Python tool to perform deep learning experiments on various hyperspectral datasets.

## Setup

This tool requires a Python environment. It should be compatible with both Python 2.7 and 3.5+.

This tool is based on [PyTorch](http://pytorch.org/). Please follow the official instructions to install PyTorch in your environment.

Remaining dependencies are specified in `requirements.txt`. They can be installed, e.g. by running:

`pip install -r requirements.txt`

Some dependencies could probably be ignored, e.g. Visdom and IPython.

## Datasets

You can either download the datasets beforehand on the [UPV/EHU](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes) wiki or let the tool download them for you.

At this time, the available datasets are the following:
  * Pavia University
  * Pavia Center
  * Kennedy Space Center
  * Indian Pines
  * Botswana

We plan to add other datasets soon.

## Models

Several models are available:
  * SVM (RBF and linear, with grid search)
  * baseline neural network (4 fully connected layers with dropout)
  * spatial-spectral CNN from "Deep Learning Approach for Remote Sensing Image Analysis", Hamida et al. (BiDS 2016)
  * 3D fully convolutional network from "Contextual Deep CNN Based Hyperspectral Classification", Lee and Kwon (IGARSS 2016)
  * 3D CNN from "Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks", Chen et al. (TGRS 2016)
  * 3D CNN from "Spectral–Spatial Classification of Hyperspectral Imagery with 3D Convolutional Neural Network", Li et al. (Remote Sensing 2017)

## Usage

Start a Visdom server:
`python -m visdom.server`
and go to http://localhost:8097 to see the visualizations.

Then, run the script `main.py`.

Alternatively, you can use the following [Jupyter notebook](https://gist.github.com/nshaud/25882091996aaf9e54a1b47ba97daf03) by downloading it in the main folder. **Note : there is no guarantee that this notebook will stay up-to-date with the latest versions of this tool. Be warned.**

The most useful arguments are:
  * `--model` to specify the model ('svm', 'nn', 'hamida', 'lee', 'chen', 'li'),
  * `--dataset` to specify which dataset to use ('PaviaC', 'PaviaU', 'IndianPines', 'KSC', 'Botswana'),
  * the `--cuda` switch to run the neural nets on GPU.

There are more parameters that can be used to control more finely the behaviour of the tool. See `python main.py -h` for more information.

Examples:
  * `python main.py --model SVM --dataset IndianPines --training_sample 0.3`
    This runs a grid search on SVM on the Indian Pines dataset, using 30% of the samples for training and the rest for testing. Results are displayed in the visdom panel.
  * `python main.py --model nn --dataset PaviaU --training_sample 0.1 --cuda`
    This runs on GPU a basic 4-layers fully connected neural network on the Pavia University dataset, using 10% of the samples for training.
  * `python main.py --model hamida --dataset PaviaU --training_sample 0.5 --patch_size 7 --epoch 50 --cuda`
    This runs on GPU the 3D CNN from Hamida et al. on the Pavia University dataset, using 50% of the samples for training and optimizing for 50 epochs.

