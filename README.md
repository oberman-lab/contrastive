# Contrastive-Inspired Semi-Supervised Learning

This repo is intended to to serve as a framework for comparing loss functions in a semi-supervised learning context. 

## TODO
 - [ ] Define architecture for image datasets (Use Chris's Implementation of LeNet from Hello World)
 - [ ] Modularize loss classes
 - [ ] Expand on paper list below

## Interesting related papers
- [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)


## Current Supported Datasets
  - Projection Dataset
  - MNIST
  - Fashion MNIST
  - CIFAR-10
  
  --------------------------------------------------------------------------------
 ### Projection Dataset:
 Inputs of size 1x2n.
 
 The first n components are a one-hot vector in R<sup>n</sup>. The remaining values are ~ N(0,1)
 
 It is expected that we will learn a matrix [I 0] where I is the R<sup>nxn</sup> identity and 0 the zero matrix in R<sup>nxn</sup>
 
 This dataset is intented to be a simple toy dataset that is easy and quick to train and has a known solution. 
 
  --------------------------------------------------------------------------------
 ### MNIST
 Inputs of size 28x28x1
 
 The classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset of handwritten decimal digits. MNIST is an easy dataset to learn. If a method doesn't work here, it probably won't work anywhere.

 --------------------------------------------------------------------------------
 ### Fashion MNIST
 Inputs of size 28x28x1
 
 The [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset meant to be a harder substitute for MNIST.
 
 This dataset is intented to be the first 'real' dataset for proof of concept training. 


 --------------------------------------------------------------------------------
### CIFAR-10
Inputs of size 32x32x3

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset of color images in 10 categories. 

The next step from Fashion MNIST with the same number of categories but much harder to classify.
