from torch import nn as nn
import torch.nn.functional as F

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



class CenterLeNet(nn.Module):  # as seen in https://kpzhang93.github.io/papers/eccv2016.pdf (center loss paper)
    def __init__(self,dropout,device):
        super(CenterLeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,5,padding=2),
            nn.Conv2d(32,32,5,padding=2), # 2x conv(5,32)
            nn.PReLU(),
            nn.MaxPool2d(2,stride=2)
        ).to(device)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,5,padding=2),
            nn.Conv2d(64,64,5,padding=2), # 2x conv(5,64)
            nn.PReLU(),
            nn.MaxPool2d(2,stride=2)
        ).to(device)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,5,padding=2),
            nn.Conv2d(128,128,5,padding=2), # 2x conv(5,128)
            nn.PReLU(),
            nn.MaxPool2d(2,stride=2)
        ).to(device)
        self.linear1 = nn.Sequential( # apply activation
            _View(1152),
            nn.Linear(1152, 2)
        ).to(device)
        self.linear2 = nn.Linear(2,10).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x
