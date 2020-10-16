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











