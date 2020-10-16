
import torch
import argparse
import pytest
import traceback
from data_processing.contrastive_data import *

''' Install pytest - navigate to this directory - call pytest - pray nothing broke'''

# Parse arguments
parser = argparse.ArgumentParser(description='Constrastive Learning Experiment')
parser.add_argument('--batch-size-labeled', type=int, default=8, metavar='NL',
                    help='input labeled batch size for training (default: 128)')

parser.add_argument('--batch-size-unlabeled', type=int, default=8, metavar='NU',
                    help='input unlabeled batch size for training (default: 128)')

parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--dropout', type=float, default=0.25, metavar='P',
                    help='dropout probability (default: 0.25)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='heavy ball momentum in gradient descent (default: 0.9)')
parser.add_argument('--frac-labeled', type=float, default=0.1, metavar='FL',
                    help='Fraction of labeled data (default 0.01))')
parser.add_argument('--num-clusters', type=int, default=5, metavar='NC',
                    help='Number of clusters to expect')
parser.add_argument('--data-dir', type=str, default='./data', metavar='DIR')
parser.add_argument('--log-interval', type=int, default=100, metavar='LOGI')
parser.add_argument('--loss-function', type=str, default='MSELoss', metavar='LF')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


''' Testing functions below here '''

def test_labelsStrippedMNIST():
    '''If the labels are stripped, the first element '''
    dataLoaders = ContrastiveData(args.frac_labeled,args.data_dir,args.batch_size_labeled,args.batch_size_unlabeled,dataset_name = 'Projection', **kwargs).get_data_loaders()
    test= iter(dataLoaders['unlabeled'])
    assert (type(next(test)) is not list), 'The labels were not stripped off'

def test_loadMNIST():
    try:
        dataLoaders = ContrastiveData(args.frac_labeled,args.data_dir,args.batch_size_labeled,args.batch_size_unlabeled,dataset_name = 'MNIST', **kwargs).get_data_loaders()

        for i,(data,label) in enumerate(dataLoaders['labeled']):
            print(label)
    except Exception:
        print(traceback.print_exc())
        pytest.fail("Loading MNIST failed")

def test_loadProjection():
    try:
        dataLoaders = ContrastiveData(args.frac_labeled,args.data_dir,args.batch_size_labeled,args.batch_size_unlabeled,dataset_name = 'Projection', **kwargs).get_data_loaders()
    except Exception:
        print(traceback.print_exc())
        pytest.fail("Loading ProjectionDataset failed")


def test_badData():
    with pytest.raises(ValueError) as context:
        dataLoaders = ContrastiveData(args.frac_labeled,args.data_dir,args.batch_size_labeled,args.batch_size_unlabeled,dataset_name = '!@##$$)*^!^@##@****!!@GDTGENOTADATASET', **kwargs)
    assert  "Dataset name is not supported" in str(context)

def test_genProjectionData():
    import shutil # Don't want this to have access outside of this function
    path = 'tempDirectoryForTesting'
    data = ProjectionData(path,train = True,num_clusters=10)
    item, label = data.__getitem__(0)
    shutil.rmtree(path)  # Removes directory to force data generation
    assert label.size() == item.size(), 'Couldn\'t generate ProjectionData'
