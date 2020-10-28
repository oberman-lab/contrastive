## Similarity and Euclidean Distance Relationship

Cosine similarity is given by 
$$
sim(a,b)= \frac{a^Tb}{\||a||\cdot ||b||}
$$
Now clearly this is pretty close to Euclidean distance. In fact
$$
d(a,b) = (a-b)^T(a-b) = a^Ta - 2a^b + b^Tb = ||a||^2 -2||a||\cdot||b|| sim(a,b) + ||b||^2
$$


If we assume that $a,b$ are normalized then 
$$
d(a,b) = 2(1-sim(a,b))
$$


or 
$$
sim(a,b) = 1 - \frac{d(a,b)}{2}
$$


and from this it is obvious that maximizing the similarity is equivalent to minimizing the distance. 



### Learning an embedding

If we assume that our category labels are given by normalized vectors (ie. $e_i$) then we see the equivalancy between distances and cosine similarities. Which one is easier to compute? Similarity requires only one $O(n)$ operation—dot product. Distance requires that plus a subtraction. 

They essentially seem the same. We should just choose the one that makes the most sense intuitively, theoretically. 

# Semi-supervised Thinking

As described in [MixMatch](https://arxiv.org/abs/1905.02249) there seems to be three broad trends incorporated into semi-supervised loss functions. 

- Consistancy regularization
- Entropy minimization
- Traditional regularization

I would like to get a better understanding of the theory behind these ideas as they seem to be built on inuition and emperical success. Better theory could lead to better implementations. 

### Consitency Regularization

This idea is that $label(Augment(x)) = label(x)$. 

As far as I can tell, there hasn't been much theory or thought into this assumption. My intuition agrees with this for small augmentations. A large augmentation could certainly violate this rule—for example, zeroing all inputs. The middle ground is where it gets interesting and building work behind this could lead to a better understanding of how this assumption should be implemented. 

One could imagine that $Augment(x)$ and $x$ are drawn from the same latent distribution. The problem is then knowing enough about these latencies to construct $Augment(\cdot)$. 

### Entropy Minimization

Categories with lower entropy should be preffered. This could be implemented as a sharpening step, minimizing the distance between obeservations with the embedding or more. 

For labeled data $x$ with embedding $z$ and label $y$ in the embedded space, one could add a term such as $d(z,y)$ to the loss. Incentivizing each data point to be close to it's label, would ensure that data in different categories should be embedded far apart. 

For unlabeled data $u$ with embedding $z_u$ this is a trickier idea. There is no natural target for the model to embed to. Currently the way we have approached this is to minimize the $d(z_u,C(z_u))$ where $C(\cdot) = \min_i\{d(z_u,y_i)\}$ returns the closest label $y_i \in Y$. 

	- Early on in training, when the embedding is parameterized with random weights this metric could lead to poor performance. I imagine we should run a handful of epochs using just the labeled data in order to get a prototype embedding first. 
	- Additionally we should smooth this approach. Replace $C(\cdot) = \min_i\{d(z_u,y_i)\}$ with $C(\cdot) = RealSoftMin_i\{d(z_u,y_i)\}$. This allows for auto-differentiation

### Traditional Regularization

By 'traditional' we refer to methods such as LASSO or Ridge Regression. These essentially make it harder for the model to memorize the labels and weights and thus overfit. By adding a fairly simple term to the loss, one can greatly improve regularization. 

I have a hunch that augmentation is doing something similiar to this. By augmenting we provide another sample that we expect will be the same as other known data. This augmentation is represented differently, yet has the same labelling. This should in effect, incentivize the model to learn the true latent distribution that we assume $Augement(x)$ and $x$ come from, rather than record known observations. 



# Notes from VanEngelen Survey

Clarifies some of the assumptions implicitly made in semi-supervised learning. Namely

- $p(x)$ contains some information about $p(y | x)$. Seems trivial but is important to remember
  - Seems to be pretty reasonable and holds in most real world applications
  - As a consequence the following subtypes of assumptions may arise:
    - (Smoothness) That given $x, x'$ 'close' to each other that $y = y'$					
    -  (Low-Density)  Decision boundary shouldn't pass through areas of high density in the input space
    - (Manifold) Data points on the same low-dimensional manifold should have the same label

### Smoothness Assumption

Assuming $x,x'$ have the same label in a semi-supervised context brings up the following implication. 

Given one labeled point $x_l$ and two unlabeled point $x_1,x_2$ where $x_1$ is close to $x_l$ and $x_2$ but $x_2$ is only close to $x_1$ then by transitivity we expect $x_l$ and $x_2$ to have the same label despite not being close. 

I imagine one could look at this pesimistically, however you gain extra information at no cost expect the assumption. 

### Low Density Assumption

The boundary should not pass through high density regions. Thus it is prefrerable that the it pass through low density regions. IE

... ..  ... .. .....|     +    +++++  +++++++++   

... ..  ... .. .....   |   +    +++++  +++++++++   

The second boundary is preferable, despite having the same classification error. This idea is closely related to the concept of margin for SVMs.

### Manifold Assumption

From the review paper

*"where the data can be represented in Euclidean space, the observed data points in the high-dimensional input space Rd are usually concentrated along lower-dimensional substructures. These substructures are known as manifolds: topological spaces that are locally Euclidean. For instance, when we consider a 3-dimensional input space where all points lie on the surface of a sphere, the data can be said to lie on a 2-dimensional manifold. The manifold assumption in semi-supervised learning states that (a) the input space is composed of multiple lower-dimensional manifolds on which all data points lie and (b) data points lying on the same manifold have the same label. Consequently, if we are able to determine which manifolds exist and which data points lie on which manifold, the class assignments of unlabelled data points*"



# Reached page 382





# Notes about MSE vs Distance

It is much easier in Pytorch to use the MSE Loss. This is built in and ready to go. For now, we've been disregarding the fact that MSE and distance are different. 

However they are closely related. Consider one labeled point $x \in \mathbb{R}^n$  and it's centre $c \in \mathbb{R}^n$ then 
$$
MSE(x,c) = \frac{1}{n} \sum_i (x_i - c_i)^2
$$
and 
$$
d(x,c) = \sum (x_i - c_i)^2
$$
Good! If we want to use distance itself, we just have scale by $n$. 



But...PyTorch expects everything in batches.

Now given a batch of points $x^{(i)} \in \mathbb{R^n}, i = 1\dots b$ with each point having an MSE of 
$$
MSE_i = MSE(x^{(i)},c^{(i)}) = \frac{1}{n} \sum_j (x_j^{(i)} - c_j^{(i)})^2
$$
 PyTorch's MSE function will return 
$$
MSE(X,C) = \frac{1}{b} \sum_i MSE_i = \frac{1}{b n } \sum_i d(x^{(i)},c^{(i)})
$$
IE. It returns the **batch mean** of the MSE. 





## Within cluster energy

If we define for a cluster $C$ with center $c$
$$
E(C) = \frac{1}{|C|} \sum d(x,c), x\in C
$$
and we wish to minimize this, then essentially all we are doing is adding an embedding layer to $k$-means. With the added benefit that by fixing the centres beforehand we don't have to actually perform $k$-means. (Maybe, we want to let centres be free in a more $k$-means-y fashion ) 



We want this embedding layer to have a few nice properties. I think Mathew summed it up nicely

















