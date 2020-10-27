## Note on Cluster energy loss for contrastive learning

For contrastive learning, Mike and Mido took advantage of small number of labels to design a new ad hoc loss, which trained better.


As I understand it, we ant to combine the ideas of

-  multiple views

- cluster energy

- using known labels

Into a different loss, which better expresses the goals of training (as I understand them).

There is a non-convex cluster energy in the unsupervised case, which measures how well clustered a set of points, by summing the distance squared to the barycenter of each cluster.

However, if we have labels for a few points, we can assume that these points are embedded to a fixed vector $c_i$, (which can be 1-hot vectors, for example).

Then we can minimize the cluster energy. 

We also want to use views to say that differnt views should classify the same.

---



Given 

- $c_i$ the centers, which are fixed (can be 1-hot vectors).

Given $x_1, \dots, x_n$ data points (some with labels, some without).  

For the unlabelled points, the classification of $x$ is given by 

- $C(x) = \arg \min d(x,c_i)^2$ , the distance squared to the nearest cluster center. 
- Define $C_i$ to be the set of the points in the cluster $i$.
- Define the ith Cluster energy by  $E_i = mean_{x\in C_i} d(x,c_i)^2$  intra-cluster variance
- and the total Cluster energy is $\sum_i E_i$ 

#### Define a soft version of the loss to minimize the cluster energy

#### For known labels

In this case, when we know $C(x) = j$ then we should penalize the distance to $c_j$ (instead of the closest one).  So $d_j(x) = d(x,c_j)$, so loss is $d_j(x)^2$. 

#### For points without known labels

Replace $d(x,c_i)^2$ with $d(x,C(x))^2$

 Problem: in the definitio of $C(x)$ there is arg min. Not differentiable (or compatible with neural networks) So replace instead with $RealSoftMin ( d(x,c_1)^2, \dots, d(x,c_n)^2)$

Log sum Exp (LSE) is the RealSoftMax is given by  $ \log ( \exp(d^2_1) + \dots \exp(d^2_n))$  So RealSoftMin is given by  $ -\log ( \exp(-d^2_1) + \dots \exp(-d^2_n))$ 

(so no need to check which cluster you get)

### What about using similarity?

$(z-c_i) = z^2 - z c_i + c_i^2 $ amnd when you normalize $= (2 - z\cdot c_i)$

given $z_i$ want to say it is similar to $z_i^+$.  We can enforce this with the term.

 $d(z_i, z_i^+)^2$  (usual similarity)

Or, in each cluster energy, we can "double" the $x$ along with $x^+$ and ask that they classify to the same place. 



---



### Punch line

Note $x$ means $f(x)$ the imbedding of $x$, the data point. 

$Term1 =  mean~not~sum \sum_{ (x_i, y_i)} \left (d(x_i,y_i = c_i(x_i))^2 + d(x^+ , y_i = c_i(x_i))^2 \right)$

for known labels, where $x^+$ is another view of $x$. 

$Term2 =    mean~not~sum \sum_{ (x_i)} \left (d(x,C(x))^2 + d(x^+,C(x))^2 \right)$ 

Note second term is $C(x)$ not $C(x^+)$ to make the view close to $x$.   Could also just be  $d(x^+,x)^2$  (maybe better).

same idea but replace the label with the presumed label, given by the classification of the model. 

In the loss we are skipping the basic term:

$d(x,x^+)^2$

but maybe it's there, implicitly...



How do other views fit it?  Their way: was to say, simply $d(x,x^+)^2$ small in the loss. Here could do that, or say, both close to classification. I don't know which is better...

They also said : " I should be far from other data points".  Is that necessary? I asked, and they did for historical reasons.  But Occam's razor: start simple. 

$$Loss  = Term1 + \lambda Term2$$

Flexibility: How do you define the views?  Do we need them for the know labels?  Maybe not, but why not make the definition symmetric?  This way we requires known data to also be invariant to views. 

#### Discussion

Could this be more efficient than the current example? Contrastive loss is working hard to say that the image should be far away from all other examples.  But this is already built into requiring that it be close to the centre. (Close to centre means far from other centers, and other point also want to be close to centre )

### Enhancement

Note, maybe I don't believe that my view is perfect.  E.g. Cropping should only worth with prob .9.

Then my loss should reflect this.  In other words.   

$Term1 =  mean~not~sum \sum_{ (x_i, y_i)} \left (d(x_i,y_i = c_i(x_i))^2 + .9* d(x^+ , y_i = c_i(x_i))^2 \right)$

Or early in training, make the view term, smaller, then ramp it up ...

### Sketch of toy example:

data in low dimensions already separated into $x,y$ where in $x$, the data is perfectly clustered, and in $y$ the data is random.  Now make your transforms move $y$ around, and keep $x$ fixed.  The map you learn is projection onto $x$. 

Next, take MNIST.  Do the method we describe, but you need a similarity transform. 

