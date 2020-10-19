import torch


def returnClosestCenter(centers, points):
    ''' Returns the value of the center closest to each point
        - input
    '''
    # thank you pytorch for excellent indexing abilities
    distance = torch.cdist(centers, points)
    m, indicies = torch.min(distance, 0)
    closest = centers[indicies, :]
    closest.requires_grad = True
    return closest