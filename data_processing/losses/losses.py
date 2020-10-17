from data_processing.losses.helpers import returnClosestCenter
from torch.nn import MSELoss


def make_semi_sup_basic_loss(centers):
    '''Take list of centers and gives a loss function with grouping based on those centers'''

    def basic_loss(unlabeled_output, labeled_output, labels):
        my_mse_loss = MSELoss()
        return my_mse_loss(labeled_output, labels) + my_mse_loss(unlabeled_output,
                                                                 returnClosestCenter(centers, unlabeled_output))

    return basic_loss

