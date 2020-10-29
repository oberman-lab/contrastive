from semisupervised.losses.helpers import returnClosestCenter
from torch.nn import MSELoss




def semi_mse_loss(centers):
    '''Take list of centers and gives a loss function with grouping based on those centers'''

    def basic_loss(unlabeled_output, labeled_output, labels):
        mse = MSELoss()
        return mse(labeled_output, labels) + mse(unlabeled_output, returnClosestCenter(centers, unlabeled_output))

    return basic_loss

def realsoftmin_loss(centers):

    def loss(unlabeled,labeled,labels):
        mse = MSELoss()

        return mse(labeled_output, labels)

    return loss
