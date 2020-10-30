import torch
import matplotlib.pyplot as plt
import torchnet as tnt
from semisupervised.data_processing.contrastive_data import ContrastiveData

def plot_model(model, epochs, data_loaders,modeldir,saveas):
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(epochs+1,3,figsize=(20,20))
        titles = ['labeled','unlabeled','test']
        color_list = ['gold','darkblue','darkred','green','steelblue','darkturquoise','orange','purple','gray','palevioletred']
        for epoch in range(epochs+1):
            model.load_state_dict(torch.load(modeldir + 'model'+str(epoch)+'.pt',map_location=torch.device('cpu')))
            for ix,dataset in enumerate(['labeled','unlabeled_with_labels','test']):
                top1 = tnt.meter.ClassErrorMeter(accuracy = True)
                for data, target, labels in data_loaders[dataset]:
                   data = data.to('cpu')
                   target = target.to("cpu")
                   output = model(data).to("cpu")
                   top1.add(-torch.cdist(output,model.centers.to("cpu")),labels)
                   for i in range(len(color_list)):
                       axes[epoch,ix].scatter(output[labels==i,:][:,0],output[labels==i,:][:,1],color=color_list[i],s=1)
                       axes[epoch,ix].scatter(model.centers[i,0].to("cpu"),model.centers[i,1].to("cpu"),color='black')
                       axes[epoch,ix].set_xlim((-2,2))
                       axes[epoch,ix].set_ylim((-2,2))
                   if epoch == 0 :
                       axes[epoch,ix].set_title(titles[ix])
                axes[epoch,ix].set_aspect('equal')
                axes[epoch,ix].set_xlabel('Accuracy %.2f'%top1.value()[0])
    plt.subplots_adjust(wspace=-.5, hspace=0.2)
    plt.savefig(saveas,bbox_inches='tight',dpi=100)


modeldir = './models/'
epochs = 5

r = 1
pi = torch.acos(torch.zeros(1)).item() * 2
t = torch.div(2*pi*torch.arange(10),10)
x = r * torch.cos(t)
y = r * torch.sin(t)
centers = torch.cat((x.view(-1,1), y.view(-1,1)), 1)

data = ContrastiveData(0.01, './data', centers, batch_size_labeled=64,
                       batch_size_unlabeled=128, dataset_name="FashionMNIST",
                       num_clusters=10)
data_loaders = data.get_data_loaders()
model = torch.load(modeldir+'modelStruct',map_location='cpu')


plot_model(model, epochs, data_loaders,modeldir,'supervised.png')
