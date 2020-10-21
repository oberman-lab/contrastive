from arg_parser import ContrastiveArgParser
import torch
import torch.optim as optim

from losses.losses import semi_mse_loss,basic_softmin_loss
from nets import *
from procedures import run_epoch, test_model, train_supervised
from data_processing.utils import *
from data_processing.contrastive_data import ContrastiveData
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter # for logging

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

    if args.log_dir != "None":
        writer = SummaryWriter(log_dir = args.log_dir)
    else:
        writer = None

    # Define the centers (targets)
    if args.dataset == "Projection":
        num_clusters = args.num_clusters
        eye = torch.eye(2 * num_clusters, 2 * num_clusters)
        centers = eye[0:num_clusters, :].to(device)
    else:
        num_clusters = 10
        centers = torch.eye(num_clusters, num_clusters).to(device)

    # Get data
    data = ContrastiveData(args.frac_labeled, args.data_dir, batch_size_labeled=args.batch_size_labeled,
                           batch_size_unlabeled=args.batch_size_unlabeled, dataset_name=args.dataset,
                           num_clusters=num_clusters, **kwargs)
    data_loaders = data.get_data_loaders()

    # Define model and accesories
    if args.dataset == 'Projection':
        model = SimpleNet(args.num_clusters,device)
    else:
        model = LeNet(args.dropout,device)

    loss_function = basic_softmin_loss(centers)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Train the semi-supervised model
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        run_epoch(model, epoch,data_loaders, optimizer, device,args ,loss_function,writer)
        test_model(model,epoch,data_loaders, MSELoss(),centers, device,writer)
        print('Wall clock time for epoch: {}'.format(time.time() - t0))

    # Train the supervised model for comparison
    if args.compare:
        # Reset model and accesories
        if args.dataset == 'Projection':
            model = SimpleNet(args.num_clusters,device)
        else:
            model = LeNet(args.dropout,device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        loss_function = MSELoss()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_supervised(model,epoch,data_loaders,optimizer,device,args,loss_function,writer)
            test_model(model,epoch,data_loaders, MSELoss(),centers, device,writer)
            print('Wall clock time for epoch: {}'.format(time.time() - t0))
