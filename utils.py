# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
import scipy.io as scio
import gzip
from six.moves import cPickle
import sys
import theano
import theano.tensor as T
import math
from sklearn.utils.linear_assignment_ import linear_assignment
import os
import sys
import csv

import warnings
warnings.filterwarnings("ignore")

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def Sampling(batch_size, latent_dim):
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    return sampling

def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    # Should be ind, though not used
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, ind    

def _load_fashion_mnist():
    """Loads the Fashion-MNIST dataset.
    Copied from keras.datasets.fashion_mnist.py

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = os.path.join('dataset', 'fashion-mnist')
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    paths = [os.path.join(dirname, f) for f in files]

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)
             
def load_data(dataset: str):    
    path = os.path.join('dataset', dataset)
    if dataset == 'mnist':
        path = os.path.join(path, 'mnist.pkl.gz')
        if path.endswith(".gz"):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')
    
        if sys.version_info.major < 3:
            (x_train, y_train), (x_test, y_test) = cPickle.load(f)
        else:
            (x_train, y_train), (x_test, y_test) = cPickle.load(f, encoding="bytes")
        
        f.close()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        X = np.concatenate((x_train,x_test))
        Y = np.concatenate((y_train,y_test))
        
    elif dataset == 'reuters10k':
        data=scio.loadmat(os.path.join(path, 'reuters10k.mat'))
        X = data['X']
        Y = data['Y'].squeeze()
        
    elif dataset == 'har':
        data=scio.loadmat(os.path.join(path, 'HAR.mat'))
        X=data['X']
        X=X.astype('float32')
        Y=data['Y']-1
        X=X[:10200]
        Y=Y[:10200]

    elif dataset == 'cifar-10':        
        data=scio.loadmat(os.path.join(path, 'cifar10_feature.mat'))
        X = data['x']           # (60k, 2048)
        Y = data['y'].squeeze() # (60k,)
        np.apply_along_axis(lambda row: (row - row.mean()) / (row.std() + 1e-8),
                            1, X)    # normalize

    elif dataset == 'fashion-mnist':
        (x_train, y_train), (x_test, y_test) = _load_fashion_mnist()
        # Respectively: (60k, 28, 28), (60k,), (10k, 28, 28), (10k,)
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        X = np.concatenate((x_train,x_test))
        Y = np.concatenate((y_train,y_test))

    elif dataset == 'cifar-100':
        data = scio.loadmat(os.path.join(path, 'cifar100_feature.mat'))
        X = data['x']           # (60k, 2048)
        Y = data['y'].squeeze() # (60k,)
        np.apply_along_axis(lambda row: (row - row.mean()) / (row.std() + 1e-8),
                            1, X)   # normalize

    elif dataset == 'svhn':
        train_data = scio.loadmat(os.path.join(path, 'train_gist.mat'))
        test_data = scio.loadmat(os.path.join(path, 'test_gist.mat'))
        X = np.concatenate((train_data['X'], test_data['X']))
        Y = np.concatenate((train_data['Y'], test_data['Y'])).squeeze()
        np.apply_along_axis(lambda row: (row - row.mean()) / (row.std() + 1e-8),
                            1, X)   # normalize
        Y = Y - Y.min()             # s.t. Y.min() = 0
        assert len(X) == len(Y) == 73257 + 26032

    elif 'bing_query' in dataset:
        path = os.path.join('dataset', 'bing_query')
        if not os.path.exists(os.path.join(path, dataset) + '.mat'):
            X = []
            Y = []
            
            with open(os.path.join(path, dataset) + '.tsv') as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t')
                for line in reader:
                    temp = line[1]
                    temp = temp.split(',')
                    if (len(temp) == 768):
                        X.append(temp)
                        Y.append(int(line[0]))
            X = np.asarray(X, dtype=np.float32)
            X = X/(np.max(X) - np.min(X))
            Y = np.asarray(Y, dtype=np.float32)

            scio.savemat(os.path.join(path, dataset),
                         {'X': X, 'Y': Y})

        else:
            data = scio.loadmat(os.path.join(path, dataset + '.mat'))
            X = data['X']
            Y = data['Y'].squeeze()

    else:
        assert False

    return X, Y 

def config_init(dataset: str, pre_train=False):
    '''original_dim, epoch, n_centroid, lr_nn, lr_gmm,
    decay_n, decay_nn, decay_gmm, alpha, datatype(activation of x_decoded_mean)'''
    if dataset == 'mnist':
        return 784, 3000 if not pre_train else 10, 10, 0.002, 0.002,\
            10, 0.9, 0.9, 1, 'sigmoid' 
    elif dataset == 'reuters10k':
        return 2000, 15 if not pre_train else 5, 4, 0.002, 0.002,\
            5, 0.5, 0.5, 1, 'linear'
    elif dataset == 'har':
        return 561, 120 if not pre_train else 10, 6, 0.002, 0.00002,\
            10, 0.9, 0.9, 5, 'linear'
    elif dataset in ('cifar-10', 'cifar-100'):
        return 2048, 120 if not pre_train else 5, int(dataset.split('-')[-1]), 0.002, 0.00002,\
            10, 0.9, 0.9, 5, 'linear'   
    elif dataset == 'fashion-mnist':
        return 784, 120 if not pre_train else 5, 10, 0.002, 0.00002,\
            10, 0.9, 0.9, 5, 'linear'   
    elif dataset == 'svhn':
        return 960, 120 if not pre_train else 5, 10, 0.002, 0.00002,\
            10, 0.9, 0.9, 5, 'linear'   
    elif dataset == 'bing_query1':
        return 768, 100 if not pre_train else 4, 10, 0.008, 0.00008,\
            10, 0.98, 0.98, 5, 'linear'  
    else:
        assert False
