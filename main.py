import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from data_processing.utils import *
from data_processing import cycle_no_memory
from data_processing.contrastive_data import ContrastiveData

# Parse arguments
parser = argparse.ArgumentParser(description='Constrastive Learning Experiment')
parser.add_argument('--batch-size-labeled', type=int, default=8, metavar='NL',
                    help='input labeled batch size for training (default: 128)')

parser.add_argument('--batch-size-unlabeled', type=int, default=8, metavar='NU',
                    help='input unlabeled batch size for training (default: 128)')

parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--dropout', type=float, default=0.25, metavar='P',
                    help='dropout probability (default: 0.25)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='heavy ball momentum in gradient descent (default: 0.9)')
parser.add_argument('--frac-labeled', type=float, default=0.1, metavar='FL',
                    help='Fraction of labeled data (default 0.01))')
parser.add_argument('--num-clusters', type=int, default=5, metavar='NC',
                    help='Number of clusters to expect')
parser.add_argument('--data-dir', type=str, default='./data', metavar='DIR')
parser.add_argument('--log-interval', type=int, default=100, metavar='LOGI')
parser.add_argument('--loss-function', type=str, default='MSELoss', metavar='LF')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

# Print out arguments to the log
print('Constrastive Learning Run')
for p in vars(args).items():
    print('  ', p[0] + ': ', p[1])
print('\n')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


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


### Let's define the simplest network I can
class SimpleNet(nn.Module):  # With Projection data we should see the identity map from R^n to R^N/2

    def __init__(self, num_clusters):
        super(SimpleNet, self).__init__()
        self.num_clusters = num_clusters
        self.net = nn.Sequential(
            nn.Linear(2 * num_clusters, 2 * num_clusters, bias=False)
        )

    def forward(self, x):  # No activation function, just one map of weights.
        return self.net(x)


MSELoss = torch.nn.MSELoss()
loss_function = MSELoss


num_clusters = args.num_clusters
eye = torch.eye(2 * num_clusters, 2 * num_clusters)
centers = eye[0:num_clusters, :]

data = ContrastiveData(args.frac_labeled, args.data_dir,batch_size_labeled=args.batch_size_labeled, batch_size_unlabeled=args.batch_size_unlabeled, dataset_name='Projection',
                       num_clusters=num_clusters, **kwargs)
data_loaders = data.get_data_loaders()

# Define model and accesories
model = SimpleNet(num_clusters)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

if args.cuda:
    model.cuda()
    centers.cuda()

def train_jointly(epoch, centers, loss_function=None, **kwargs):
    model.train()
    # We loop over all batches in the (bigger) unlabeled set. While we do so we loop also on the labeled data, starting over if necessary.
    # This means that unlabeled data may be present many times in the same epoch.
    # The number of times that the labeled data is processed is dependent on the batch size of the labeled data.
    #for batch_ix, unlabeled_features in enumerate(data_loaders['unlabeled']):
    for batch_ix, (unlabeled_features, (labeled_features, labels)) in enumerate(cycle_with(leader=data_loaders['unlabeled'], follower=data_loaders['labeled'])):
        if args.cuda:
            unlabeled_features, labeled_features, labels, centers = unlabeled_features.cuda(), labeled_features.cuda(), labels.cuda(), centers.cuda()

        labeled_output = model(labeled_features)
        unlabeled_output = model(unlabeled_features)
        loss = loss_function(unlabeled_output, labeled_output, labels, centers)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_ix % args.log_interval == 0 and batch_ix > 0:
            print('[Epoch %2d, batch %3d] training loss: %.4f' %
                  (epoch, batch_ix, loss))

def basic_loss(unlabeled_output, labeled_output, labels, centers):
    return MSELoss(labeled_output, labels) + MSELoss(unlabeled_output, returnClosestCenter(centers, unlabeled_output))


initial_weights = model.net[0].weight.clone()
def test():
    model.eval()
    test_loss = tnt.meter.AverageValueMeter()
    with torch.no_grad():
        for data, target in data_loaders['test']:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = loss_function(output, target)
            test_loss.add(loss.cpu())
    print('[Epoch %2d] Average test loss: %.5f'
          % (epoch, test_loss.value()[0]))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_jointly(epoch, centers, loss_function=basic_loss)
        print(time.time() - t0)
        test()
