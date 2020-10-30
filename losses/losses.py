from losses.helpers import returnClosestCenter
from torch.nn import MSELoss

def semi_center_loss(centers):
    '''Take list of centers and gives a loss function with grouping based on those centers'''
    
    def basic_loss(unlabeled_output, output, labels):
        mse = MSELoss()
        return mse(output, labels) + mse(unlabeled_output, returnClosestCenter(centers, unlabeled_output))
        
        # return mse(output_features, labels_features) + mse(output_ll, labels_ll) + mse(unlabeled_output_features, returnClosestCenter(centers_features, unlabeled_output_features)) + mse(unlabeled_output_ll, returnClosestCenter(centers_ll, unlabeled_output_ll))
    
    return basic_loss

def center_loss():
    '''Take list of centers and gives a loss function with grouping based on those centers'''
    
    def basic_loss(output, labels):
        mse = MSELoss()
        return mse(output, labels)
    
    return basic_loss

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
