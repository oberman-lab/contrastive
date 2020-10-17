from data_processing.contrastive_arg_parser import ContrastiveArgParser
import torch
import torch.optim as optim
import torchnet as tnt

from data_processing.losses import make_semi_sup_basic_loss
from data_processing.nets import SimpleNet
from data_processing.utils import *
from data_processing.contrastive_data import ContrastiveData
from torch.nn import MSELoss


def run_epoch(model, current_epoch, optimizer, device, loss_function=None):
    model.train()
    # We loop over all batches in the (bigger) unlabeled set. While we do so we loop also on the labeled data, starting over if necessary.
    # This means that unlabeled data may be present many times in the same epoch.
    # The number of times that the labeled data is processed is dependent on the batch size of the labeled data.
    # for batch_ix, unlabeled_features in enumerate(data_loaders['unlabeled']):
    for batch_ix, (unlabeled_features, (labeled_features, labels)) in enumerate(
            cycle_with(leader=data_loaders['unlabeled'], follower=data_loaders['labeled'])):

        unlabeled_features = unlabeled_features.to(device)
        labeled_features = labeled_features.to(device)
        labels = labels.to(device)

        labeled_output = model(labeled_features)
        unlabeled_output = model(unlabeled_features)
        loss = loss_function(unlabeled_output, labeled_output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_ix % args.log_interval == 0 and batch_ix > 0:
            print('[Epoch %2d, batch %3d] training loss: %.4f' %
                  (current_epoch, batch_ix, loss))


def test_model(model, loss_function, device):
    # Define model and accesories
    model.eval()
    test_loss = tnt.meter.AverageValueMeter()
    with torch.no_grad():
        for data, target in data_loaders['test']:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            test_loss.add(loss.cpu())
    print('[Epoch %2d] Average test loss: %.5f'
          % (epoch, test_loss.value()[0]))


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

    num_clusters = args.num_clusters
    eye = torch.eye(2 * num_clusters, 2 * num_clusters).to(device)
    centers = eye[0:num_clusters, :]

    data = ContrastiveData(args.frac_labeled, args.data_dir, batch_size_labeled=args.batch_size_labeled,
                           batch_size_unlabeled=args.batch_size_unlabeled, dataset_name='Projection',
                           num_clusters=num_clusters, **kwargs)
    data_loaders = data.get_data_loaders()

    model = SimpleNet(num_clusters, device)
    loss_function = make_semi_sup_basic_loss(centers)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        run_epoch(model, epoch, optimizer, device, loss_function=loss_function)
        test_model(model, MSELoss(), device)
        print(time.time() - t0)
