from semisupervised.arg_parser import ContrastiveArgParser
import torch
import torch.optim as optim
from semisupervised.losses.losses import semi_mse_loss
from semisupervised.nets import *
from semisupervised.procedures import run_epoch, test_model, train_supervised, plot_model
from semisupervised.data_processing.utils import *
from semisupervised.data_processing.contrastive_data import ContrastiveData
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
        r = 1
        pi = torch.acos(torch.zeros(1)).item() * 2
        t = torch.true_divide(2*pi*torch.arange(10),10)
        x = r * torch.cos(t)
        y = r * torch.sin(t)
        centers = torch.cat((x.view(-1,1), y.view(-1,1)), 1).to(device)
#        centers = torch.eye(num_clusters, num_clusters).to(device)

    # Get data
    data = ContrastiveData(args.frac_labeled, args.data_dir, centers, batch_size_labeled=args.batch_size_labeled,
                           batch_size_unlabeled=args.batch_size_unlabeled, dataset_name=args.dataset,
                           num_clusters=num_clusters, **kwargs)
    data_loaders = data.get_data_loaders()

    # Define model and accesories
    if args.dataset == 'Projection':
        model = SimpleNet(args.num_clusters,device)
    else:
        model = LeNet2D(args.dropout,device,centers)
#        model = LeNet(args.dropout,device)

    loss_function = semi_mse_loss(centers)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Train the semi-supervised model
    torch.save(model.state_dict(), 'model'+str(0)+'.pt')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        run_epoch(model, epoch,data_loaders, optimizer, device,args ,loss_function,writer)
        test_model(model,epoch,data_loaders, MSELoss(), device,writer)
        torch.save(model.state_dict(), 'model'+str(epoch)+'.pt')
        print('Wall clock time for epoch: {}'.format(time.time() - t0))

    plot_model(model, args.epochs, data_loaders, device, 'cluster_semi.png')

    # Train the supervised model for comparison
    if args.compare:
        # Reset model and accesories
        if args.dataset == 'Projection':
            model = SimpleNet(args.num_clusters,device)
        else:
            model = LeNet2D(args.dropout,device,centers)
#            model = CenterLeNet(args.dropout,device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        loss_function = MSELoss()


        torch.save(model.state_dict(), 'model'+str(0)+'.pt')
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_supervised(model,epoch,data_loaders,optimizer,device,args,loss_function,writer)
            test_model(model,epoch,data_loaders, MSELoss(), device,writer)
            torch.save(model.state_dict(), 'model'+str(epoch)+'.pt')
            print('Wall clock time for epoch: {}'.format(time.time() - t0))

        plot_model(model, args.epochs, data_loaders, device,'cluster_supervised.png')

    #torch.save(model,'CenterLeNet_saved')
