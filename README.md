# Automatic detection of Ionospheric Alfvén Resonances using U-net

This repository was maintained by Paolo Marangio as part of the Dissertation project counting towards the degree of Master of Science in High Performance Computing with Data Science. The work presented here is the result of a collaboration between the Edinburgh Parallel Computing Center (EPCC) and the British Geological Survey (BGS).

---

## Overview

### Data

The data about the IARs phenomenon has been collected by BGS over the past 7 years using high frequency induction coils installed at Eskdalamuir Observatory. This has been devided into 178 days used for training and 2135 days used for testing. You can find these images(size 701x1101) in folder data/membrane.

### Model

![img/u-net-architecture.png](img/u-net-architecture.png)

This repository has been forked from the repository made by https://github.com/zhixuhao/unet. Following that, the codebase was adpated according to our needs.

In order to achieve the task of segmentation on images displaying IARs signal, the fully convolutional neural network U-net is implemented with Keras functional API.

This is the reference to the original paper describing the U-net architecture [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

Output from the network is a 256x256 image that represents the segmentation mask generated for a given test image.

### Training

The model is trained for 10 Epochs on IARs training data with binary crossentropy as loss function and Intersection over Union(IoU) as evaluation metric. 

FINAL IOU SCORE TO BE ADDED
FINAL TRAINING LOSS


### Code features

TO BE ADDED

### Branches overview

TO BE ADDED

---

## How to use

### Dependencies

The programme depends on the following libraries:

* Tensorflow
* Keras
* Numpy
* Scikit-image
* Matplotlib

A full list of program dependencies can be found in requirements_pip.txt and requirements_conda.txt.

This code should be compatible with Python versions 2.7-3.5.

### Run main.py

The programme can be run on either CPU or GPU. Given small dataset size, 1 GPU was optimal. 

You will see the predicted results of test image in data/membrane/test

### Results

#Use the trained model to do segmentation on test images, the result is statisfactory.

TO BE ADDED

#![img/0test.png](img/0test.png)

#![img/0label.png](img/0label.png)

## Overview of code used to generate Tables and Figures in Dissertation manuscript

| Figure (F) and/or Table (T)  | Commit        | Branch  | 
| -------------        | ------------- | ------- |
| F5.1         |TBA   |TBA         |
| F5.2, T5.1         | f28b02bef9b04e335270e1864a1726309d5541d0  | vanilla x unet  |
| F5.2, T5.1         | 18788411c90fe0630299c6c61413a6a42d6f1fe7  | vanilla x unet  |
|F5.2, T5.1         | adf7e58961d4d0d4b6ba54ff3efbc42cef8cf01f  | vanilla x unet  |
|F5.2, T5.1         | 3d438e1ee001a40fc7f3aeb78a19cf45ede35f26  | vanilla x unet  |
|          |   |         |
|          |   |         |
|          |   |         |
|          |   |         |


## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Keras has the following features:

- allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
- supports both convolutional networks and recurrent networks, as well as combinations of the two.
- supports arbitrary connectivity schemes (including multi-input and multi-output training).
- runs seamlessly on CPU and GPU.
Documentation can be found here [Keras.io](http://keras.io/)



