import torch
import torchnet as tnt

from data_processing import cycle_with

def run_epoch(model, current_epoch, data_loaders, optimizer, device, args, loss_function=None):
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


def test_model(model,current_epoch, data_loaders, loss_function, device):
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
          % (current_epoch, test_loss.value()[0]))