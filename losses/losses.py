from losses.helpers import returnClosestCenter
from torch.nn import MSELoss



def semi_mse_loss(centers):
    '''Take list of centers and gives a loss function with grouping based on those centers'''

    def basic_loss(unlabeled_output, labeled_output, labels):
        my_mse_loss = MSELoss()
        return my_mse_loss(labeled_output, labels) + my_mse_loss(unlabeled_output,
                                                                 returnClosestCenter(centers, unlabeled_output))

    return basic_loss

# def basic_softmin_loss(centers):
#     pdist = PairwiseDistance(p=2)
#
#     def loss(unlabeled_output, labeled_output, labels):
#         S_l = labeled_output.size()[0]
#         S_u = unlabeled_output.size()[0]
#
#         labeled_loss = torch.sum(pdist(labeled_output,labels)) / S_l
#         #unlabeled_loss = torch.sum(-torch.logsumexp(-torch.cdist(unlabeled_output,centers),dim=-1)) / S_u
#         unlabeled_loss = torch.sum(pdist(unlabeled_output,returnClosestCenter(centers,unlabeled_output)))  / S_u
#
#         return labeled_loss + unlabeled_loss
#
#
#
#
#
#     return loss
