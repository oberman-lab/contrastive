from torch import nn as nn


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
                                 nn.Conv2d(ci,co,ksz,stride=1,padding=2),
                                 nn.BatchNorm2d(co),
                                 nn.ReLU(True),
                                 nn.Conv2d(co,co,ksz,stride=1,padding=2),
                                 nn.BatchNorm2d(co),
                                 nn.ReLU(True),
                                 nn.MaxPool2d(psz,stride=psz),
                                 nn.Dropout(p))
        
        self.m = nn.Sequential(
                               convbn(1,32,5,2,dropout),
                               convbn(32,64,5,2,dropout),
                               convbn(64,128,5,2,dropout),
                               _View(128*3*3),
                               nn.Linear(128*3*3,2),
                               nn.Linear(2,10)).to(device)
    
    def forward(self, x):
        return self.m(x)

class LeNetplus(nn.Module):
    def __init__(self,dropout,device,centers):
        super(LeNetplus, self).__init__()
        
        self.centers = centers
        
        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                                 nn.Conv2d(ci,co,ksz,stride=1,padding=2),
                                 nn.BatchNorm2d(co),
                                 nn.ReLU(True),
                                 nn.Conv2d(co,co,ksz,stride=1,padding=2),
                                 nn.BatchNorm2d(co),
                                 nn.ReLU(True),
                                 nn.MaxPool2d(psz,stride=psz),
                                 nn.Dropout(p))
        
        self.m = nn.Sequential(
                               convbn(1,32,5,2,dropout),
                               convbn(32,64,5,2,dropout),
                               convbn(64,128,5,2,dropout),
                               _View(128*3*3),
                               nn.Linear(128*3*3,2)).to(device)
#        self.ll = nn.Linear(2,10).to(device)

#    def forward(self, x):
#        return self.ll(self.m(x))

#    def features(self, x):
#        return self.m(x)

#    def last_layer(self, x):
#        return self.ll(x)
    def forward(self, x):
        return self.m(x)
