import torch
import torchnet as tnt
from data_processing import cycle_with
from losses.helpers import returnClosestCenter
import matplotlib.pyplot as plt



def run_epoch(model, current_epoch, data_loaders, optimizer, device, args,loss_function,writer):
    model.train()
    # We loop over all batches in the (bigger) unlabeled set. While we do so we loop also on the labeled data, starting over if necessary.
    # This means that unlabeled data may be present many times in the same epoch.
    # The number of times that the labeled data is processed is dependent on the batch size of the labeled data.
    # for batch_ix, unlabeled_features in enumerate(data_loaders['unlabeled']):
    for batch_ix, (unlabeled_images, (labeled_images, labels)) in enumerate(
            cycle_with(leader=data_loaders['unlabeled'], follower=data_loaders['labeled'])):
        unlabeled_images = unlabeled_images.to(device)
        labeled_images = labeled_images.to(device)
        labels = labels.to(device)

        output = model(labeled_images)
        unlabeled_output = model(unlabeled_images)
        loss = loss_function(unlabeled_output, output, model.centers[labels,:])

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
        loss = loss_function(output, model.centers[labels,:])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_ix % (args.log_interval//5) == 0 and batch_ix > 0:
            print('Fully Supervised: [Epoch %2d, batch %3d] training loss: %.4f' %
                  (current_epoch, batch_ix, loss))
    if writer is not None:
        writer.add_scalar('train/loss/fully', loss,current_epoch)

def test_model(model,current_epoch, data_loaders, loss_function, device,writer):
    model.eval()
    top1 = tnt.meter.ClassErrorMeter(accuracy = True)
    test_loss = tnt.meter.AverageValueMeter()
    with torch.no_grad():
        for data, target in data_loaders['test']:
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            top1.add(-torch.cdist(output,model.centers),target)
            loss = loss_function(output, model.centers[target,:])
            test_loss.add(loss.cpu())

    print('[Epoch %2d] Average test loss: %.5f, Accuracy: %.2f'
          % (current_epoch, test_loss.value()[0], top1.value()[0]))
    if writer is not None:
        writer.add_scalar('test/loss',  test_loss.value()[0],current_epoch)
        writer.add_scalar('test/acc', top1.value()[0],current_epoch)



def plot_model(model, epochs, data_loaders, device, saveas):
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(epochs,3,figsize=(20,20))
        titles = ['labeled','unlabeled','test']
        color_list = ['gold','darkblue','darkred','green','steelblue','darkturquoise','orange','purple','gray','palevioletred']
        for epoch in range(epochs+1):
            model.load_state_dict(torch.load('model'+str(epoch)+'.pt',map_location=torch.device('cpu')))
            for ix,dataset in enumerate(['labeled','unlabeled_with_labels','test']):
                top1 = tnt.meter.ClassErrorMeter(accuracy = True)
                for data, target in data_loaders[dataset]:
                    data = data.to(device)
                    target = target.to("cpu")
                    output = model(data).to("cpu")
                    top1.add(-torch.cdist(output,model.centers.to("cpu")),target)
                    for i in range(len(color_list)):
                        axes[epoch,ix].scatter(output[target==i,:][:,0],output[target==i,:][:,1],color=color_list[i],s=1)
                        axes[epoch,ix].scatter(model.centers[i,0].to("cpu"),model.centers[i,1].to("cpu"),color='black')
                        axes[epoch,ix].set_xlim((-2,2))
                        axes[epoch,ix].set_ylim((-2,2))
                    if epoch == 0 :
                        axes[epoch,ix].set_title(titles[ix])
                    axes[epoch,ix].set_aspect('equal')
                    axes[epoch,ix].set_xlabel('Accuracy %.2f'%top1.value()[0])
        plt.subplots_adjust(wspace=-.5, hspace=0.2)
        plt.savefig(saveas,bbox_inches='tight',dpi=100)

