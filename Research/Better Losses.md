# Better Losses

### Center Loss

The center loss was proposed in [1] and extended in [2]. The goal is to encourage intra-class compacteness by encouraging the deep features to cluster around centers. Previous approaches used contrastive losses and triplet losses which require a dramatic increase of the number of training pairs or triples and ultimately slower convergence and instability. This was done in the context of face recognition.

The center loss was used in [3] to show improvement to adversarial attacks.

### OLÉ Loss

One of the disadvantages of the center loss is that it needs to be used with the standard cross-entropy loss which encourages inter-class separation. This was addressed in [4] with the introduction of the OLÉ loss at the expanse of an increase in training time between 10% and 33%. Unlike [1] and [2], [4] focuses more on image classification. Results show marginal improvements on MNIST, CIFAR10, CIFA100 and STL-10. The most interesting result is the latter since it is a dataset designed for self-supervised and semi-supervised learning.

### Mahalanobis center (MMC) Loss

According to [5], when using the center loss together with the cross entropy loss, there is a trade-off between inter-class dispersion and intra-class compactness. They propose the MMC loss which uses only the center loss with the centers being pre-computed based on the Max-Mahalonobis distribution. This has the added benefit of not having to update the centers during training as in the case of the center loss. The focus of the paper was in adversarial robustness and it is a clear improvement on the center loss.

### Large-Margin Softmax Loss

In the earlier paper [6], the large-margin softmax loss is proposed in order to increase the inter-class dispersion.

### Contrastive loss

The contrastive loss was originally proposed in [7]. It was extended to the supervised context in [8]. A combination with center loss is proposed in [9].


#### References

[1] [A Discriminative Feature Learning Approach for Deep Face Recognition - Wel et al (2016) ECCV](https://kpzhang93.github.io/papers/eccv2016.pdf)

[2] [A Comprehensive Study on Center Loss for Deep Face Recognition - Wen et al (2019) IJCV](https://link.springer.com/article/10.1007/s11263-018-01142-4)

[3] [Improving Robustness to Adversarial Examples by Encouraging Discriminative Features - Agarwal et al (2019) ICIP](https://ieeexplore.ieee.org/document/8803601)

[4] [OLE - Orthogonal Low-rank Embedding, A Plug and Play Geometric Loss for Deep Learning - Lezama et al (2018) CVPR](https://ieeexplore.ieee.org/document/8578944)

[5] [Rethinking Softmax Cross-Entropy Loss for Adversarial Robustness - Pang et al (2020) ICLR](https://openreview.net/forum?id=Byg9A24tvB)

[6] [Large-Margin Softmax Loss for Convolutional Neural Networks - Liu et al (2020) ICML](http://proceedings.mlr.press/v48/liud16.html)

[7] [Dimensionality Reduction by Learning an Invariant Mapping - Hadsell et al (2006) CVPR](https://ieeexplore.ieee.org/document/1640964)

[8] [Supervised Contrastive Learning - Khosla et al (2020)](https://arxiv.org/abs/2004.11362)

[9] [Contrastive-center Loss for Deep Neural Networks - Ce Qi and Fei Su (2017) ICIP](https://ieeexplore.ieee.org/document/8296803)
