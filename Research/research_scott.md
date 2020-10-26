# Contrastive-Inspired Semi-Supervised Learning

This repo is intended to to serve as a framework for comparing loss functions in a semi-supervised learning context. 

## Sources To Read

### Papers
- [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)
- [A Survey on Semi-Supervised Learning](https://link.springer.com/article/10.1007/s10994-019-05855-6)
- [Contrastive Representation Learning: A Framework and Review](https://arxiv.org/abs/2010.05113)
- [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf)

### Wikipedia
- [Wikipedia: Semi-supervised learning](https://en.wikipedia.org/wiki/Semi-supervised_learning)
- [Wikipedia: K-means Clustering](https://en.m.wikipedia.org/wiki/K-means_clustering)

### Local Files
See the Files folder

## Coding
See the main README

## Takeways
Disclaimer: this section is about reporting from sources as well as my own understanding and insights from the sources.
As such anything written below should only be accepted after some critical thought.
### Wikipedia: Semi-Supervised Learning
#### Supervised Learning: Transductive learning - Generalisation p(y|x)
#### Unsupervised Learning: Inductive learning - Inductions from patterns in p(x) to get p(y|x)
##### Assumptions
- Continuity Assumption: points close together are likely to share labels -> motivates putting boundaries 
in low density areas. Intuition: this is about probabilistic lipschitzness of $p(y|x)$ weighted on the density p(x)
- Cluster Assumption: Data tends to cluster. With continuity assumption, this means clusters share labels, which motivates
feature learning by cluster.
- Manifold Assumption: Most of the dimensions are redundant. All usefull information can be projected onto a lower-dimensional
manifold. Assumption motivated by the scientific method: learning from observation is to reduce the complexity of the theory.

##### Methods
- Generative Models: model all of the information. These methods focus on finding p(x|y). Once it is known, p(y|x) is then
infered using Baye's rule: p(y|x) = p(x|y)p(y)/p(x). Philosophy of this method: guess what the elements of set looked like before
it got mixed with other sets (ex: in the task of separating cats from dogs, focus on what cats look like, and what dog looks like).
 Each of Sup, Unsup learning finds some of the information related to p(x|y).
Generative modeling finds more information then either sup or unsup learning alone.
Unsup Learning learns about p(x), and then (with help of labels) tries to decompose the function $p(x) = \sum_i p(x|y_i)p(y_i).
Sup Learning finds max(p(y|x)) directly-by learning a hypothesis function h(x)=y, and then would have to additionnaly find measures
of confidence in order to model p(x|y).
Such models learn $p(x|y,\theta)$, and make assumptions about which p(x|y) are possible (choice of a hypothesis class).
Wrong assumptions can hurt(negative impact! not neutral. source: Gweon et al, 2010, "Infants consider both the sample and the sampling process in inductive generalization").
 Good assumptions necessarily helps performance (source: Younger B. A. (1999) "Parsing Items into Separate Categories: Developmental Change in Infant Categorization" ).
 Example of generative model: Gaussian mixture distribution.
 Used by fitting: Kayle divergence on both the joint distribution and the unlabeled distribution
 $$\argmax_{\theta}(\sum_i\log p({x_i, y_i}|theta) + \lambda \sum_i \logp(\{x_i\}|\theta))$$
 
 - Low-density Separation:
 Idea: Place boundaries in low density regions. Ex: TSVM transductive support vector machine. SVM's in unsup. learning: maximize margins,
 but indiscriminantly of labels. Hinge lloss: (1-yf(x)). Unsup equivalent: (1-|f(x)|). Loss is made with sum of both losses with
 weighting $\lambda_1$, with a $\lambda_2$-weighted normalisation on the RKHS norm of h, for f(x)= h(x) + b. Other processes for low-density estimation exists.
 - Graph-Base methods:
 Idea: build a graph by proximity in x, and then model low-dim manifold by fitting fn f(x) by optimising smoothness of f to the manifold. The smoothness of manifold
 is optimized, and smoothness over input space as well
 
 #### Wikipedia: k-means
 ##### Idea
  We have a number k of centers, each associated with a partition of the data. Minimize sum of euclidian distance squared from
  nodes in a set to the center. For any given set, the point that does this is the mean. Therefore, minimize the partitionning
  of fixed size that minimizes distance squared to the mean of the subset.
  This is equivalent to minimizing the normalised sum of distance squared for every pairs of elements in the set. This
  equivalence is proved through the factored-out expression for $C_l$ the partition elements, 
  and $\mu_l = \frac{1}{|C_l|}\sum_{x \in C_l} x$ the mean of the partition element.
  $$\sum_{x \in C_l} ||x||^2 - |C_l| ||\mu_l||^2$$
  Proof: [proof](Files/kmeans_equivalence_proof.pdf)
  Points:
  - Non-convex function, so no guarantee of finding global max
  - Choice of k crucial and difficult. For this, one can try out many or use external information (ex: numbers of classes)
  However,# of classes is not always the best choice, if classes are not clustered well. The best choice of k is the number of distinct
  clusters.
  - Can be generalised to gaussian mixtures. K means is Gaussian mixtures for unit diagonal gaussians. 
  - Features can be learned by defining a feature as the nearest center of a point.
  
 #### Notes Oberman
  - Inspired from mido's contrastive learning paper. The idea of the paper is to combine contrastive learning on labeled
  and unlabeled data. This is done through cosine similarity between feature representations.
  
  - unsupervised: data to learn features with similarity between elements and their data augmented counterparts.
  - supervised: optimize similarity between elements of the same label.
  
  Once this pre-training is done, the network is trained with the labeled data only.
  
  Principle is to learn a representation feature vector with an approriate metric
  between them. It is also important that the feature vectors retain information about the input space.
  
  - Question: Why is cosine similarity used?
  Oberman takeway:
  - multiple views: Data augmentation -> unsupervised similarity
  - using known labels: For Mido, labels used twice: to learn distances and to learn labeling function
  - cluster energy: Absent from Mido's paper! new input.\
  
  First Idea: embed all labeled elements to fixed "target" vectors.
  ISSUE: We cannot learn then! The function from elements to labeled centers is the trivial realizable hypothesis.
  
  Fix the centers $c_i$ corresponding to the labels (can be the basis vectors).
  
  Minimize d(x, C(x)) where C(x) is the center nearest to x (unsup sample)
  ISSUE: contrary to k-means, minimizing this does not pitch distance of elements against each other.
  When centroid goes closer to other elements, it leaves others behind (compromise between samples).
  This is not the case here as the gradient in distance with different points is independent (modulo complexity of the hypothesis class).
  Result: The gradients will be determined by initial config. C(x) will (almost) never change for training,
  not learning clusters, but taking the initial clustering as true and as needed to be enforced in features.
  
  Remark: key difference between mido and this: this algorithm attempts to learn the entire problem (labels determined by distance)
  Whereas position in mido's feature learning is not decided(definition of pre-training to some extent)
  
  Mido:
  learn local metric. Then transform this space to the answers(logit space)
  
  Oberman:
  Learn both at once.
  
  Fixing label vectors equivalent to the logit space -> so it is done for all neural nets already.
  So all arguments d(x,e_i) is standard supervised learning. Arguments d(x,c(x)) means running unsupervised
  samples throught the net, finding the (untrained) prediction and then training the network to make this
  prediction more confident. The terms $d(x,x^+)$ is the pre-processing discussed earlier. The same can be
  said of d(x, y) for x and y sharing labels.
  
  
