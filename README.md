# Contrastive-Inspired Semi-Supervised Learning

This repo is intended to to serve as a framework for comparing loss functions in a semi-supervised learning context. 

## Organization

- Slides to Matthew's introductory talk on Semi-supervised learning (10/22): [Matthew's talk](https://github.com/oberman-lab/contrastive/blob/master/Research/Matthew's%20Presentation%20SSL%20-%202020-10-12.pdf)
- Brief overview (with references) to recently proposed losses, including center loss (distance to center targets): [Better Losses](https://github.com/oberman-lab/contrastive/blob/master/Research/Better%20Losses.md)
- Recent references on self-supervised and semi-supervised learning: [References](https://github.com/oberman-lab/contrastive/blob/master/Research/References.md)
- Scott's notes: [pdf](https://github.com/oberman-lab/contrastive/blob/master/Research/notes_semi_sup.pdf.pdf) and [markdown](https://github.com/oberman-lab/contrastive/blob/master/Research/reasearch_scott.md).
- Scott's notes on a loss framework (work in progress): [pdf](https://github.com/oberman-lab/contrastive/blob/master/Research/Loss_Framework.pdf)
- Adam's notes on cluster energy: [markdown](https://github.com/oberman-lab/contrastive/blob/master/Research/Cluster%20energy.md)
- Noah's general thoughts [markdown](https://github.com/oberman-lab/contrastive/blob/master/Research/noahnotes.md)
- [Blog post](https://amitness.com/2020/07/semi-supervised-learning/) that reviews some recent methods for semi-supervised learning.

## TODO
 - [ ] Add data augmentation capabilities
 - [ ] Implement more robust, detailed logging
 - [ ] Expand on paper list below
 - [ ] Visualization for Toy example (projection); use to compare losses longterm.
 - [ ] Brainstorm Several Losses for comparison
 - [x] Define architecture for image datasets (Adapt Chris's Implementation of LeNet from Hello World for MNIST)
 - [x] Modularize loss classes
 - [x] UML (ish) diagram
 - [X] More robust splitting into labeled unlabeled - ie frac-labeled = 0.8 crashes rn

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
