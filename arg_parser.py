import argparse


class ContrastiveArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description='Constrastive Learning Experiment')
        self.add_argument('--batch-size-labeled', type=int, default=32, metavar='NL',
                          help='input labeled batch size for training (default: 32)')
        self.add_argument('--batch-size-unlabeled', type=int, default=128, metavar='NU',
                          help='input unlabeled batch size for training (default: 128)')
        self.add_argument('--epochs', type=int, default=5, metavar='N',
                          help='number of epochs to train (default: 5)')
        self.add_argument('--compare', type=bool, default=False, metavar='CO',
                          help='Train supervised model for comparison?')                          
        self.add_argument('--lr', type=float, default=0.1, metavar='LR',
                          help='learning rate (default: 0.1)')
        self.add_argument('--dropout', type=float, default=0.25, metavar='P',
                          help='dropout probability (default: 0.25)')
        self.add_argument('--momentum', type=float, default=0.9, metavar='M',
                          help='heavy ball momentum in gradient descent (default: 0.9)')
        self.add_argument('--frac-labeled', type=float, default=0.01, metavar='FL',
                          help='Fraction of labeled data (default 0.01))')
        self.add_argument('--num-clusters', type=int, default=5, metavar='NC',
                          help='Number of clusters to expect')
        self.add_argument('--dataset', type=str, default='Projection', metavar='DS',
                          help='What dataset to use')
        self.add_argument('--data-dir', type=str, default='./data', metavar='DIR')
        self.add_argument('--log-interval', type=int, default=100, metavar='LOGI')
        self.add_argument('--loss-function', type=str, default='MSELoss', metavar='LF')
