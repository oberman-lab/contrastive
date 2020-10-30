# Design Experiments
## Center Losses in semi-supervised
### Idea
Compare contrastive losses vs center losses in the context of optimizing accuracy in classification.

Perform feature learning in the feature learning/ fine tuning paradigm with:
 - Contrastive losses
 - pseudo-classification losses: Create one class for every image in batch, train classification for X epochs on
 classifying images in the batch.
 - pseudo-classification losses with center loss
 then compare for accuracy.
 
 ### Motivation
 Reduce the number of training samples while keeping all the information. from Augmented $\times$ Augmented to Augmented $\times mnum_images,
 which is a compression by a factor of {number_of_augmentations}.
 ### Network
 Encoder: Convolutional for mnist
 Classifier: linear classifier
 
 ### Data Feeding
 - Implement stochastic data augmentation.
 - Create a dataset where each sample is a bundle of data augmentations, together with a weight vector in feature space.
 
 ### Process
 - add sub-training classification routine.
 
 ### Extensions to the experiment
 Add a projection head, aka do the same but discard the last layer.