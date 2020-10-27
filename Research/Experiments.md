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
 
 ### Network
 Encoder: Convolutional for mnist
 Classifier: linear classifier
 
 ### Extensions to the experiment
 Add a projection head, aka do the same but discard the last layer.