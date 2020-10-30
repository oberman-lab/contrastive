from semisupervised.arg_parser import ContrastiveArgParser
import torch
import torch.optim as optim
from semisupervised.losses.losses import semi_mse_loss
from semisupervised.nets import *
from semisupervised.procedures import run_epoch, test_model, train_supervised, plot_model,getTSNE
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
        model = SimpleNet(args.num_clusters,centers,device)
    else:
        num_clusters = 10
        if args.model == "LeNet2D":
            r = 1
            pi = torch.acos(torch.zeros(1)).item() * 2
            t = torch.div(2*pi*torch.arange(10),10)
            x = r * torch.cos(t)
            y = r * torch.sin(t)
            centers = torch.cat((x.view(-1,1), y.view(-1,1)), 1).to(device)
            model = LeNet2D(args.dropout,centers,device)
        elif args.model == "LeNet":
            centers = torch.eye(num_clusters, num_clusters).to(device)
            model = LeNet(args.dropout,centers,device)


    # Get data
    data = ContrastiveData(args.frac_labeled, args.data_dir, centers, batch_size_labeled=args.batch_size_labeled,
                           batch_size_unlabeled=args.batch_size_unlabeled, dataset_name=args.dataset,
                           num_clusters=num_clusters, **kwargs)
    data_loaders = data.get_data_loaders()


    loss_function = semi_mse_loss(centers,lam = 1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    if args.track:
        tsne_dict = {} # For visualizing
        nsamples = 5000

    # Train the semi-supervised model
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if args.track:
            tsne_dict[epoch-1]=getTSNE(model,epoch,data_loaders,nsamples,device)
            torch.save(model.state_dict(), './models/model'+str(epoch-1)+'.pt')

        run_epoch(model, epoch,data_loaders, optimizer, device,args ,loss_function,writer)
        test_model(model,epoch,data_loaders, MSELoss(), device,writer)

        print('Wall clock time for epoch: {}'.format(time.time() - t0))
    if args.track:
        tsne_dict[epoch]=getTSNE(model,epoch,data_loaders,nsamples,device)
        torch.save(model.state_dict(), './models/model'+str(epoch)+'.pt')

    # Train the supervised model for comparison
    if args.compare:
        # Reset model and accesories
        if args.dataset == "Projection":
            model = SimpleNet(args.num_clusters,centers,device)
        else:
            num_clusters = 10
            if args.model == "LeNet2D":
                model = LeNet2D(args.dropout,centers,device)
            elif args.model == "LeNet":
                model = LeNet(args.dropout,centers,device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        loss_function = MSELoss()

        if args.track:
            torch.save(model.state_dict(), './models/model'+str(0)+'.pt')
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_supervised(model,epoch,data_loaders,optimizer,device,args,loss_function,writer)
            test_model(model,epoch,data_loaders, MSELoss(), device,writer)
            if args.track:
                torch.save(model.state_dict(), './models/model'+str(epoch)+'.pt')
            print('Wall clock time for epoch: {}'.format(time.time() - t0))

    if args.track:
        torch.save(tsne_dict,'TSNE_dict')
