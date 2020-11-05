from torch import nn as nn
import torch

class SimpleNet(nn.Module):  # With Projection data we should see the identity map from R^n to R^N/2
    '''Let's define the simplest network I can'''

    def __init__(self, num_clusters, device):
        super(SimpleNet, self).__init__()
        self.num_clusters = num_clusters
        self.net = nn.Sequential(
            nn.Linear(2 * num_clusters, 2 * num_clusters, bias=False)
        ).to(device)

    def forward(self, x):  # No activation function, just one map of weights.
        return self.net(x)


class _View(nn.Module):
    def __init__(self,o):
        super(_View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)


class LeNet(nn.Module):

    def __init__(self,dropout,device):
        super(LeNet, self).__init__()

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1,20,5,3,dropout),
            convbn(20,50,5,2,dropout),
            _View(50*2*2),
            nn.Linear(50*2*2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(500,10)).to(device)

    def forward(self, x):
        return self.m(x)

class Encoder(nn.Module):
    def __init__(self, dropout, device):
        super(Encoder, self).__init__()

        def convbn(ci, co, ksz, psz, p):
            return nn.Sequential(
                nn.Conv2d(ci, co, ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz, stride=psz),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1, 20, 5, 3, dropout),
            convbn(20, 50, 5, 2, dropout),
            _View(50 * 2 * 2),
            nn.Linear(50 * 2 * 2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(500, 100)).to(device)

    def forward(self, x):
        return self.m(x)


class VisualHeadFirst(nn.Module):

    def __init__(self, device, batch_size):
        super(VisualHeadFirst, self).__init__()
        self.m = nn.Linear(100, 2).to(device)

    def forward(self, x):
        return self.m(x)

class VisualHeadSecondCosined(nn.Module):

    def __init__(self, device, batch_size):
        super(VisualHeadSecondCosined, self).__init__()
        #self.m = torch.nn.functional.normalize(torch.rand(1,batch_size, 2), dim=1).repeat(batch_size, 1,1)
        #self.m.requires_grad_(True)
        #self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.m = nn.Linear(2, batch_size, bias=False).to(device)

    def re_norm_weights(self):
        with torch.no_grad():
            self.m.weight.div_(torch.norm(self.m.weight, dim=1, keepdim=True))

    def forward(self, x):
        normed_x = nn.functional.normalize(x,dim=1)
        return self.m(normed_x)


class VisualHeadSecond(nn.Module):

    def __init__(self, device, batch_size):
        super(VisualHeadSecond, self).__init__()
        #self.m = torch.nn.functional.normalize(torch.rand(1,batch_size, 2), dim=1).repeat(batch_size, 1,1)
        #self.m.requires_grad_(True)
        #self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.m = nn.Linear(2, batch_size, bias=False).to(device)

    def forward(self, x):
        return self.m(x)


class Fine_Tuning_Head(nn.Module):

    def __init__(self, device):
        super(Fine_Tuning_Head, self).__init__()

        self.m = nn.Sequential(
            nn.Linear(100, 100),
            nn.Linear(100, 10)
        ).to(device)

    def forward(self, x):
        return self.m(x)


class Feature_Learning(nn.Module):

    def __init__(self, Encoder, Visual_Head, device):
        super(Feature_Learning, self).__init__()
        self.m = nn.Sequential(
            Encoder,
            Visual_Head
        ).to(device)

    def forward(self, x):
        return self.m(x)


class Final_Classifier(nn.Module):

    def __init__(self, Encoder, Classifier, device):
        super(Final_Classifier, self).__init__()
        self.m = nn.Sequential(
            Encoder,
            Classifier
        ).to(device)

    def forward(self, x):
        return self.m(x)

