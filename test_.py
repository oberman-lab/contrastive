import torch
import argparse
import pytest
import traceback
from arg_parser import ContrastiveArgParser
from data_processing.contrastive_data import *

''' Install pytest - navigate to this directory - call pytest - pray nothing broke'''

parser = ContrastiveArgParser()
args = parser.parse_args()
args.cuda =False #False until proven otherwise

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    args.cuda = True
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
''' Testing functions below here '''



def test_labelsStrippedMNIST():
    '''If the labels are stripped, the first element '''
    dataLoaders = ContrastiveData(args.frac_labeled, args.data_dir, args.batch_size_labeled, args.batch_size_unlabeled,
                                  dataset_name='Projection', **kwargs).get_data_loaders()
    test = iter(dataLoaders['unlabeled'])
    assert (type(next(test)) is not list), 'The labels were not stripped off'


def test_loadMNIST():
    try:
        dataLoaders = ContrastiveData(args.frac_labeled, args.data_dir, args.batch_size_labeled,
                                      args.batch_size_unlabeled, dataset_name='MNIST', **kwargs).get_data_loaders()

        for i,(data,label) in enumerate(dataLoaders['labeled']):
            print(label)

    except Exception:
        print(traceback.print_exc())
        pytest.fail("Loading MNIST failed")


def test_loadFashionMNIST():
    try:
        dataLoaders = ContrastiveData(args.frac_labeled,args.data_dir,args.batch_size_labeled,args.batch_size_unlabeled,dataset_name = 'Fashion-MNIST', **kwargs).get_data_loaders()

        for i,(data,label) in enumerate(dataLoaders['labeled']):
            print(label)
    except Exception:
        print(traceback.print_exc())
        pytest.fail("Loading Fashion MNIST failed")

def test_CIFAR10():
    try:
        dataLoaders = ContrastiveData(args.frac_labeled,args.data_dir,args.batch_size_labeled,args.batch_size_unlabeled,dataset_name = 'CIFAR10', **kwargs).get_data_loaders()
        for i,(data,label) in enumerate(dataLoaders['labeled']):
            print(label)
    except Exception:
        print(traceback.print_exc())
        pytest.fail("Loading Fashion MNIST failed")


def test_loadProjection():
    try:
        dataLoaders = ContrastiveData(args.frac_labeled, args.data_dir, args.batch_size_labeled,
                                      args.batch_size_unlabeled, dataset_name='Projection', **kwargs).get_data_loaders()
    except Exception:
        print(traceback.print_exc())
        pytest.fail("Loading ProjectionDataset failed")


def test_badData():
    with pytest.raises(ValueError) as context:
        dataLoaders = ContrastiveData(args.frac_labeled, args.data_dir, args.batch_size_labeled,
                                      args.batch_size_unlabeled, dataset_name='!@##$$)*^!^@##@****!!@GDTGENOTADATASET',
                                      **kwargs)
    assert "Dataset name is not supported" in str(context)


def test_genProjectionData():
    import shutil  # Don't want this to have access outside of this function
    path = 'tempDirectoryForTesting'
    data = ProjectionData(path, train=True, num_clusters=10)
    item, label = data.__getitem__(0)
    shutil.rmtree(path)  # Removes directory to force data generation
    assert label.size() == item.size(), 'Couldn\'t generate ProjectionData'


def test_cycle_with():
    from data_processing.utils import cycle_with
    unlabeled = [1, 2, 3, 4, 5, 6, 7]
    labeled = [1, 2, 3]
    result = [(1, 1), (2, 2), (3, 3), (4, 1), (5, 2), (6, 3), (7, 1)]
    check = []
    for elt in cycle_with(unlabeled, labeled):
        check.append(elt)
    assert check == result, "Iterator not working"
