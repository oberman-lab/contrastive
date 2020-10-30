import torch
import matplotlib.pyplot as plt


dict = torch.load('./TSNE_dict')
for epoch, (tsne,labels) in dict.items():
    if epoch >= 6:
        break

    plt.subplot(2,3,epoch+1)
    plt.scatter(tsne[:,0],tsne[:,1],c=labels)
    plt.colorbar()
    plt.title('TNSE epoch {}'.format(epoch))


plt.show()
