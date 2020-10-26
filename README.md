# Contrastive-Inspired Semi-Supervised Learning

This repo is intended to to serve as a framework for comparing loss functions in a semi-supervised learning context. 

## TODO
 - [ ] Define architecture for image datasets (Adapt Chris's Implementation of LeNet from Hello World for MNIST)
 - [ ] Modularize loss classes
 - [ ] More robust splitting into labeled unlabeled - ie frac-labeled = 0.8 crashes rn
 - [ ] Add data augmentation capabilities
 - [ ] UML (ish) diagram
 - [ ] Implement more robust, detailed logging
 - [ ] Expand on paper list below
 - [ ] Visualization for Toy example (projection); use to compare losses longterm.
 - [ ] Brainstorm Several Losses for comparison

## Interesting related papers
- [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)
- [A Survey on Semi-Supervised Learning](https://link.springer.com/article/10.1007/s10994-019-05855-6)
- [Contrastive Representation Learning: A Framework and Review](https://arxiv.org/abs/2010.05113)
- [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf)
- [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170)


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
