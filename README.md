# DeepHyperX

A Python tool to perform deep learning experiments on various hyperspectral datasets.

## Setup

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

Start a Visdom server and run the script.

See `python main.py -h` for more information.
