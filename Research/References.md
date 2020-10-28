# References Semi-supervised Learning

The main reference on [Matthew's talk](https://github.com/oberman-lab/semi-supervised/blob/main/Matthew's%20Presentation%20SSL%20-%202020-10-12.pdf) is 
- [A survey on semi-supervised learning - van Engelen, J.E. and Hoos, H.H. (2020)](https://doi.org/10.1007/s10994-019-05855-6)

It is based on the following textbooks:
- Introduction to Semi-Supervised Learning Xiaojin Zhu and Andrew B. Goldberg (2009)
- Semi-supervised Learning O Chapelle, B Scholkopf, A Zien (2009).

## Recent references on Semi-supervised Learning
 - [MixMatch A Holistic Approach to Semi-Supervised Learning - David Berthelot et al (2019) NIPS](http://papers.nips.cc/paper/8749-mixmatch-a-holistic-approach-to-semi-supervised-learning.pdf)
 - [ReMixMatch: Semi-Supervised Learning with Distribution Matching and Augmentation Anchoring - David Berthelot et al (2020) ICLR](https://openreview.net/forum?id=HklkeR4KPB)
 - [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence - Sohn et al (2020) NIPS](https://arxiv.org/pdf/2001.07685.pdf)

The MixMatch does not introduce any new insights in the field. Their main contribution is in simplifying and unifing different ideas. Their method can be reduced to: (i) use MixUp to augment the datasets of both labeled and unlabeled images; (ii) for labeled images, use the standard cross entropy loss; (iii) for unlabeled images use l2-norm between the guess of the model and a pseudo label obtained by quering the model K=2 times and sharpening the prediction.

ReMixMatch improves on MixMatch by introducing Augmentation Anchoring. Using stronger augmentation in MixMatch makes it unstable on the unlabeled images, so they counter that by setting the pseudo label as the query of the model of a weakly augmented unlabeled image and comparing it to the model query of K strongly augmented unlabeled images. Notice that their K predictions are no longer averaged and the comparison is no longer done with the l2-norm, but with the standard cross entropy. Moreover, the pseudo label is sharpened as in MixMatch and in addition they also do distribution alignment. They also had the rotation loss which they borrow straight from the self-supervised setting.

FixMatch is essentially a simplification of ReMixMatch, where they remove the rotation loss, distribution alignment and sharpening. Instead the loss on unlabeled images is only active when the pseudo-label is above a certain threshold (in other words, only when the model is confident).

Other interesting references
- [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf)
- [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170)

# References Contrastive Learning

Perhaps the big breakthrough in contrastive learning is the paper
 - [1] [A Simple Framework for Contrastive Learning of Visual Representations - Chen et al (2020) ICML](https://proceedings.icml.cc/static/paper_files/icml/2020/6165-Paper.pdf)

This is the paper that [Mido et al (2020)](https://arxiv.org/pdf/2006.10803.pdf) are trying to improve on. While the original paper is on self-supervised learning, the insight here is to bring semi-supervised learning into contrastive learning with the introduction of the SuNCet loss.


The same authors of [1] have already improved upon it:
 - [Big Self-Supervised Models are Strong Semi-Supervised Learners - Chen et al (2020)](https://arxiv.org/pdf/2006.10029.pdf)
 
Their main contributions boil down to the following
1. Train larger models
2. Increase the capacity of the projection head
3. Incorporate the memory mechanism from [MoCo](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)

# References on Self-supervised Learning

In self-supervised learning the goal is to learn image features without knowing any of the labels, however it is easily confused with semi-supervised learning due to the way one accesses the quality of the representations learned. This is done by attaching a linear layer to the encoder learned and training the linear layer only (i.e. the encoder stays fixed) to solve a classification problem (CIFAR-10, Imagenet) where now the labels of the images are known. This is known as linear evaluation. Another possibility is fine-tuning where the task is the same but the weight of the encorder are no longer frozen.

Given that no labels are known, the idea here is to define a pretext task to learn the features.
- [Unsupervised Representation Learning by Predicting Image Rotations - Giradis et al (2018) ICRL](https://openreview.net/pdf?id=S1v4N2l0-)

Here the goal was to train the neural net to predict if the image had been rotated by 0, 90, 180, and 270 degrees. Simple idea that worked well at the time. For comparison with a neural net with the same number of parameters, the contrastive learning proposed in [1] improves the results from ~55% to ~75% Top-1 accuracy on Imagenet with linear evaluation.
