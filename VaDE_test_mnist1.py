# -*- coding: utf-8 -*-
"""
VaDE (Variational Deep Embedding:A Generative Approach to Clustering)
mnist clustering min loss 93.2964: 
@code author: Zhuxi Jiang and Vishal Srivastava
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
import scipy.io as scio
import gzip
from six.moves import cPickle
import sys
import  theano
import  theano.tensor as T
import math
from keras.models import model_from_json
import csv
from sklearn import mixture
from sklearn.manifold import TSNE
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "9"
os.environ['KERAS_BACKEND'] = 'theano'

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
    
def sampling(args):
    z_mean, z_slog_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
#=====================================
def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size,ind
             
#==================================================
def load_data(dataset):
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

    elif dataset == 'bing_query':
        X = []
        Y = []
        with open('Query0.1Percent.tsv') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for line in reader:
                temp = line[1]
                temp = temp.split(',')
                #temp = np.asarray(temp, dtype=np.float32)
                if (len(temp) == 768):
                    #print(len(temp))
                    X.append(temp)
                    Y.append(int(line[0]))
        X = np.asarray(X, dtype=np.float32)
        X = X/(np.max(X) - np.min(X))
        Y = np.asarray(Y, dtype=np.float32)

    else:
        assert False

    return X, Y 


        
def gmm_para_init():
    
    gmm_weights=scio.loadmat('trained_model_weights/mnist_weights_gmm.mat')
    u_init=gmm_weights['u']
    lambda_init=gmm_weights['lambda']
    theta_init=np.squeeze(gmm_weights['theta'])
    
    theta_p=theano.shared(np.asarray(theta_init,dtype=theano.config.floatX),name="pi")
    u_p=theano.shared(np.asarray(u_init,dtype=theano.config.floatX),name="u")
    lambda_p=theano.shared(np.asarray(lambda_init,dtype=theano.config.floatX),name="lambda")
    return theta_p,u_p,lambda_p

#==========================

#================================
def get_gamma(tempz):
    temp_Z = T.transpose(K.repeat(tempz,n_centroid),[0,2,1])
    temp_u_tensor3 = T.repeat(u_p.dimshuffle('x',0,1),batch_size,axis=0)
    temp_lambda_tensor3 = T.repeat(lambda_p.dimshuffle('x',0,1),batch_size,axis=0)
    temp_theta_tensor3 = theta_p.dimshuffle('x','x',0)*T.ones((batch_size,latent_dim,n_centroid))
    
    temp_p_c_z = K.exp(K.sum((K.log(temp_theta_tensor3)-0.5*K.log(2*math.pi*temp_lambda_tensor3)-\
                       K.square(temp_Z-temp_u_tensor3)/(2*temp_lambda_tensor3)),axis=1))
    return temp_p_c_z/K.sum(temp_p_c_z,axis=-1,keepdims=True)
#=====================================================
def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.savefig('mnist_latentspace_tsne.png')
    return f, ax, sc
#======================================================    
ispretrain = True
batch_size = 100
latent_dim = 10
intermediate_dim = [500,500,2000]
theano.config.floatX='float32'
dataset  = 'mnist'
X,Y = load_data(dataset)
original_dim = 784
n_centroid = 10 
X = X[:len(X) - len(X) % batch_size]
Y = Y[:len(X)]
theta_p, u_p, lambda_p = gmm_para_init()

#=====================================================
#Keras Model
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim[0], activation='relu')(x)
h = Dense(intermediate_dim[1], activation='relu')(h)
h = Dense(intermediate_dim[2], activation='relu')(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
x_decoded_mean = Dense(original_dim, activation='sigmoid')(h_decoded)
p_c_z = Lambda(get_gamma, output_shape=(n_centroid,))(z_mean)

p_c_z_output = Model(x, p_c_z)
encoder = Model(x, z)
vade = Model(x, x_decoded_mean)

vade.load_weights('trained_model_weights/mnist_weights_nn.h5')


#print ('MNIST dataset VaDE - clustering accuracy: %.2f%%'%(accuracy*100))

