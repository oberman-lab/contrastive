import torch
import torchnet as tnt
from data_processing import cycle_with
from losses.helpers import returnClosestCenter



def run_epoch(model, current_epoch, data_loaders, optimizer, device, args,loss_function ,writer):
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
            print('Semi Supervised: [Epoch %2d, batch %3d] training loss: %.4f' %
                  (current_epoch, batch_ix, loss))
    if writer is not None:
        writer.add_scalar('train/loss/semi', loss,current_epoch)

def train_supervised(model,current_epoch,data_loaders,optimizer,device,args,loss_function,writer = None):
    model.train()

    for batch_ix,(data,labels) in enumerate(data_loaders['labeled']):
        data = data.to(device)
        labels = labels.to(device)

        output = model(data)
        loss = loss_function(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_ix % (args.log_interval//5) == 0 and batch_ix > 0:
            print('Fully Supervised: [Epoch %2d, batch %3d] training loss: %.4f' %
                  (current_epoch, batch_ix, loss))
    if writer is not None:
        writer.add_scalar('train/loss/fully', loss,current_epoch)

def test_model(model,current_epoch, data_loaders, loss_function,centers, device,writer):
    model.eval()
    top1 = tnt.meter.ClassErrorMeter(accuracy = True)
    test_loss = tnt.meter.AverageValueMeter()
    with torch.no_grad():
        for data, target in data_loaders['test']:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            
            top1.add(torch.cdist(output,centers),torch.argmax(target,dim = 1))
            loss = loss_function(output, target)
            test_loss.add(loss.cpu())

    print('[Epoch %2d] Average test loss: %.5f, Accuracy: %.5f'
          % (current_epoch, test_loss.value()[0], top1.value()[0]))
    if writer is not None:
        writer.add_scalar('test/loss',  test_loss.value()[0],current_epoch)
        writer.add_scalar('test/acc', top1.value()[0],current_epoch)
