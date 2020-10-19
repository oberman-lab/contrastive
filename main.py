from arg_parser import ContrastiveArgParser
import torch
import torch.optim as optim

from losses.losses import make_semi_sup_basic_loss
from nets import *
from procedures import run_epoch, test_model
from data_processing.utils import *
from data_processing.contrastive_data import ContrastiveData
from torch.nn import MSELoss

if __name__ == "__main__":
    # Parse arguments
    parser = ContrastiveArgParser()

    args = parser.parse_args()
    args.cuda =False #False until proven otherwise

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        args.cuda = True

    # Print out arguments to the log
    print('Contrastive Learning Run')
    for p in vars(args).items():
        print('  ', p[0] + ': ', p[1])
    print('\n')
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    # Define the centers (targets)
    num_clusters = args.num_clusters
    eye = torch.eye(2 * num_clusters, 2 * num_clusters)
    centers = eye[0:num_clusters, :].to(device)

    # Get data
    data = ContrastiveData(args.frac_labeled, args.data_dir, batch_size_labeled=args.batch_size_labeled,
                           batch_size_unlabeled=args.batch_size_unlabeled, dataset_name='MNIST',
                           num_clusters=num_clusters, **kwargs)
    data_loaders = data.get_data_loaders()

    # Define model and accesories
    model = LeNet(args.dropout,device)
    loss_function = make_semi_sup_basic_loss(centers)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        run_epoch(model, epoch,data_loaders, optimizer, device,args , loss_function=loss_function)
        test_model(model,epoch,data_loaders, MSELoss(), device)
        print('Wall clock time for epoch: {}'.format(time.time() - t0))
