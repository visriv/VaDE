# -*- coding: utf-8 -*-
'''
VaDE (Variational Deep Embedding:A Generative Approach to Clustering)

Best clustering accuracy: 
MNIST: 94.46% +
Reuters10k: 81.66% +
HHAR: 85.38% +
Reuters_all: 79.38% +

@code author: Zhuxi Jiang
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'theano'
import argparse
import numpy as np
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
import scipy.io as scio
import gzip
from six.moves import cPickle
import sys
import  theano
import  theano.tensor as T
import math
from sklearn import mixture
from sklearn.cluster import KMeans
from keras.models import model_from_json
from utils import *
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
        
def gmmpara_init():
    theta_init=np.ones(n_centroid)/n_centroid
    u_init=np.zeros((latent_dim,n_centroid))
    lambda_init=np.ones((latent_dim,n_centroid))
    
    theta_p=theano.shared(np.asarray(theta_init,dtype=theano.config.floatX),name="pi")
    u_p=theano.shared(np.asarray(u_init,dtype=theano.config.floatX),name="u")
    lambda_p=theano.shared(np.asarray(lambda_init,dtype=theano.config.floatX),name="lambda")
    return theta_p,u_p,lambda_p

#================================
def get_gamma(tempz):
    temp_Z=T.transpose(K.repeat(tempz,n_centroid),[0,2,1])
    temp_u_tensor3=T.repeat(u_p.dimshuffle('x',0,1),batch_size,axis=0)
    temp_lambda_tensor3=T.repeat(lambda_p.dimshuffle('x',0,1),batch_size,axis=0)
    temp_theta_tensor3=theta_p.dimshuffle('x','x',0)*T.ones((batch_size,latent_dim,n_centroid))
    
    # LYL: +1e-10 is not consistent with VaDE_test*.py, thus resulting in
    # a HUGE difference between training accuracy and testing accuracy!!
    # HOOOOOLY SH*T!!
    temp_p_c_z=K.exp (K.sum((K.log(temp_theta_tensor3)-0.5*K.log(2*math.pi*temp_lambda_tensor3)-\
                       K.square(temp_Z-temp_u_tensor3)/(2*temp_lambda_tensor3)),axis=1)) + 1e-10
    return temp_p_c_z/K.sum(temp_p_c_z,axis=-1,keepdims=True)
#=====================================================
def vae_loss(x, x_decoded_mean):
    Z=T.transpose(K.repeat(z,n_centroid),[0,2,1])
    z_mean_t=T.transpose(K.repeat(z_mean,n_centroid),[0,2,1])
    z_log_var_t=T.transpose(K.repeat(z_log_var,n_centroid),[0,2,1])
    u_tensor3=T.repeat(u_p.dimshuffle('x',0,1),batch_size,axis=0)
    lambda_tensor3=T.repeat(lambda_p.dimshuffle('x',0,1),batch_size,axis=0)
    theta_tensor3=theta_p.dimshuffle('x','x',0)*T.ones((batch_size,latent_dim,n_centroid))
    
    # +1e-10??
    p_c_z=K.exp(K.sum((K.log(theta_tensor3)-0.5*K.log(2*math.pi*lambda_tensor3)-\
                       K.square(Z-u_tensor3)/(2*lambda_tensor3)),axis=1)) +1e-10 
    # LYL: dunno why but with 1e-10 added, MNIST works well, 
    # while others typically deteriorate.
    #p_c_z += {'mnist': 1e-10, 'svhn': 1e-10, 'cifar-100': 1e-10}.get(dataset, 0)
    
    gamma=p_c_z/K.sum(p_c_z,axis=-1,keepdims=True)
    gamma_t=K.repeat(gamma,latent_dim)
    
    if datatype == 'sigmoid':
        loss=alpha*original_dim * objectives.binary_crossentropy(x, x_decoded_mean)\
        +K.sum(0.5*gamma_t*(latent_dim*K.log(math.pi*2)+K.log(lambda_tensor3)\
        +K.exp(z_log_var_t)/lambda_tensor3+K.square(z_mean_t-u_tensor3)/lambda_tensor3),axis=(1,2))\
        -0.5*K.sum(z_log_var+1,axis=-1)\
        -K.sum(K.log(K.repeat_elements(theta_p.dimshuffle('x',0),batch_size,0))*gamma,axis=-1)\
        +K.sum(K.log(gamma)*gamma,axis=-1)
    else:
        loss=alpha*original_dim * objectives.mean_squared_error(x, x_decoded_mean)\
        +K.sum(0.5*gamma_t*(latent_dim*K.log(math.pi*2)+K.log(lambda_tensor3)\
        +K.exp(z_log_var_t)/lambda_tensor3+K.square(z_mean_t-u_tensor3)/lambda_tensor3),axis=(1,2))\
        -0.5*K.sum(z_log_var+1,axis=-1)\
        -K.sum(K.log(K.repeat_elements(theta_p.dimshuffle('x',0),batch_size,0))*gamma,axis=-1)\
        +K.sum(K.log(gamma)*gamma,axis=-1)
        
    return loss
#================================

def load_pretrain_weights(vade: Model, dataset: str):
    with open(os.path.join('pretrain_weights', 'ae_'+dataset+'.json')) as file:
        ae = model_from_json(file.read())
    ae.load_weights(os.path.join('pretrain_weights', 'ae_' + dataset + '_weights.h5'))

    print('AE summary:')
    ae.summary()
    print('\nVaDE summary:')
    vade.summary()
    print()
    has_input_layer = len(ae.layers[0].get_weights()) == 0
    print('Has input layer? {}'.format(has_input_layer))

    for i in range(4):
        vade.layers[i + 1].set_weights(
            ae.layers[i + 1 if has_input_layer else i].get_weights())
    for i in range(-1, -5, -1): # -1, ..., -4
        vade.layers[i].set_weights(ae.layers[i].get_weights())

    print ('Pretrain weights loaded!')
    return vade

def gmmpara_load(dataset):
    data = scio.loadmat(os.path.join('trained_model_weights', 
                     dataset + '_weights_gmm.mat'))
    theta_init=data['theta']
    u_init=data['u']
    lambda_init=data['lambda']
    
    theta_p=theano.shared(np.asarray(theta_init,dtype=theano.config.floatX),name="pi")
    u_p=theano.shared(np.asarray(u_init,dtype=theano.config.floatX),name="u")
    lambda_p=theano.shared(np.asarray(lambda_init,dtype=theano.config.floatX),name="lambda")
    return theta_p,u_p,lambda_p

def set_cluster(dataset: str):
    sample = encoder.predict(X, batch_size=batch_size)
    if dataset == 'reuters10k':
        k = KMeans(n_clusters=n_centroid)
        k.fit(sample)
        u_p.set_value(floatX(k.cluster_centers_.T))
    else:
        g = mixture.GaussianMixture(n_components=n_centroid, covariance_type='diag',
                                    random_state=np.random.RandomState())
        g.fit(sample)
        u_p.set_value(floatX(g.means_.T))
        lambda_p.set_value((floatX(g.covariances_.T)))

def decoder_init(dataset):
    decoder_weights = scio.loadmat('trained_model_weights/' + dataset + '_decoder.mat')
    u_decoder = decoder_weights['u']
    lambda_decoder = decoder_weights['lambda']
    theta_decoder = np.squeeze(decoder_weights['theta'])
    decoder = model_from_json(open('trained_model_weights/' + dataset + '_decoder.json').read())
    decoder.load_weights('trained_model_weights/' + dataset + '_decoder_nn.h5')
    return decoder,theta_decoder,u_decoder,lambda_decoder

def get_posterior(z,u,l,sita):
    z_m=np.repeat(np.transpose(z),n_centroid,1)
    posterior=np.exp(np.sum((np.log(sita)-0.5*np.log(2*math.pi*l)-\
                       np.square(z_m-u)/(2*l)),axis=0))
    return posterior/np.sum(posterior,axis=-1,keepdims=True)

def mnist_decoder():
    index=np.asarray(ind)[:,1]
    mnist_nice_png=np.zeros((280,280))
    for i in range(10):
        k=np.where(index==i)[0][0]
        u=d_u[:,k]
        l=d_lambda[:,k]
        sample_n=10
        count=0
        while count<sample_n:
            z_sample=np.random.multivariate_normal(u,np.diag(l),(1,))
            p=get_posterior(z_sample,d_u,d_lambda,d_theta)[k]
            if p>0.999:
                img=decoder.predict(z_sample).reshape((28,28))*255.0
                mnist_nice_png[i*28:(i+1)*28,count*28:(count+1)*28]=img
                count+=1
    return np.asarray(mnist_nice_png,dtype=np.uint8)   

def scatter(x, colors, dataset):
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
    plt.savefig('plots\\' + dataset + '_latent_space_tsne.png')
    return f, ax, sc

#===================================
def lr_decay():
    if dataset == 'mnist':
        adam_nn.lr.set_value(floatX(max(adam_nn.lr.get_value()*decay_nn, 0.0002)))
        adam_gmm.lr.set_value(floatX(max(adam_gmm.lr.get_value()*decay_gmm, 0.0002)))
    else:
        adam_nn.lr.set_value(floatX(adam_nn.lr.get_value()*decay_nn))
        adam_gmm.lr.set_value(floatX(adam_gmm.lr.get_value()*decay_gmm))
    print ('lr_nn:%f'%adam_nn.lr.get_value())
    print ('lr_gmm:%f'%adam_gmm.lr.get_value())
    
def epochBegin(epoch):
    if epoch % decay_n == 0 and epoch:
        lr_decay()
    
    gamma = gamma_output.predict(X,batch_size=batch_size)
    '''
    acc = cluster_acc(np.argmax(gamma,axis=1),Y)
    global accuracy
    accuracy+=[acc[0]]
    '''
    pred = p_c_z_output.predict(X, batch_size=batch_size)
    '''
    p_c_z_acc, _ = cluster_acc(np.argmax(pred, axis=1), Y)
    #print ('acc_gmm_on_z:%0.8f'%acc_g[0])
    print('acc_p_c_z(z): %0.8f' % acc[0]) 
    print('acc_p_c_z(z_mean): {:.8f}'.format(p_c_z_acc))
    '''
    if epoch==1 and dataset == 'har' and acc[0]<0.77:
        print ('=========== HAR dataset:bad init!Please run again! ============')
        sys.exit(0)
        
class EpochBegin(Callback):
    def on_epoch_begin(self, epoch, logs):
        epochBegin(epoch)

class PreTrainCallback(Callback):
    def on_epoch_begin(self, epoch, logs):
        # Copied from commented code in epochBegin
        sample = encoder.predict(X,batch_size=batch_size)
        g = mixture.GaussianMixture(n_components=n_centroid,covariance_type='diag')
        g.fit(sample)
        p = g.predict(sample)
        acc_g, _ = cluster_acc(p,Y)
        print('acc_g: {}'.format(acc_g))

# With pre-train: 
# 1) python3 VaDE.py dataset -m pre-train
# 2) python3 VaDE.py dataset 
# Without pre-train:
# 1) python3 VaDE.py dataset -m raw-train
parser = argparse.ArgumentParser(description='VaDE training / pre-training')
parser.add_argument('dataset', default='mnist',
                    choices=['mnist', 'reuters10k', 'har', 
                             'cifar-10', 'fashion-mnist', 'cifar-100', 'svhn' ,'bing_query', 'bing_query1'],
                    help='specify dataset')
parser.add_argument('-m', '--mode', default='train',
                    choices=['train', 'pre-train', 'raw-train'],
                    help='training with or without pre-training ' \
                        + '/ pre-training (default train)')
parser.add_argument('-e', '--epoch', type=int, help='number of epochs')
parser.add_argument('-infer', '--inference', type=bool, default = True, help='latent space visualization')



args = parser.parse_args()
dataset = args.dataset
print ('{} on {}'.format(args.mode.capitalize(), dataset))
if args.epoch is not None:
    print('Epochs: {}'.format(args.epoch))

batch_size = 100
latent_dim = 10
intermediate_dim = [500,500,2000,400]
theano.config.floatX='float32'
RS = 20150101
accuracy=[]
X,Y = load_data(dataset)
print('X.shape: ' + str(X.shape))
print('Y.shape: ' + str(Y.shape))
print('Y.min()={}, Y.max()={}\n'.format(Y.min(), Y.max()))
original_dim,epoch,n_centroid,lr_nn,lr_gmm, \
    decay_n,decay_nn,decay_gmm,alpha,datatype = config_init(
        dataset, args.mode == 'pre-train')
if args.epoch is not None:
    epoch = args.epoch
theta_p,u_p,lambda_p = gmmpara_init()
#===================

X = X[:len(X) - len(X) % batch_size]
Y = Y[:len(X)]
if args.mode != 'pre-train':    # train or raw-train
    assert args.mode in ('train', 'raw-train')
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    h = Dense(intermediate_dim[3], activation='relu')(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(Sampling(batch_size, latent_dim), 
               output_shape=(latent_dim,))([z_mean, z_log_var])
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-4], activation='relu')(h_decoded)
    x_decoded_mean = Dense(original_dim, activation=datatype)(h_decoded)

    Gamma = Lambda(get_gamma, output_shape=(n_centroid,))(z)
    p_c_z = Lambda(get_gamma, output_shape=(n_centroid,))(z_mean)

    #========================
    encoder = Model(x, z)
    gamma_output = Model(x, Gamma)
    p_c_z_output = Model(x, p_c_z)    
    vade = Model(x, x_decoded_mean)
    
    set_cluster(dataset)
    adam_nn= Adam(lr=lr_nn, epsilon=1e-4)
    adam_gmm= Adam(lr=lr_gmm, epsilon=1e-4)
    vade.compile(optimizer=adam_nn, loss=vae_loss,
                 add_trainable_weights=[theta_p,u_p,lambda_p],
                 add_optimizer=adam_gmm)
    
    epoch_begin=EpochBegin()

    if args.mode == 'train':
        #load_pretrain_weights(vade, dataset)
        vade.load_weights(os.path.join('trained_model_weights', 
                                   dataset + '_nn.h5'))
        theta_p,u_p,lambda_p = gmmpara_load(dataset)
    #-------------------------------------------------------

    vade.fit(X, X,
            shuffle=True,
            nb_epoch=epoch,
            batch_size=batch_size,   
            callbacks=[epoch_begin])

    vade.save_weights(os.path.join('trained_model_weights', 
                                   dataset + '_nn.h5'))
    scio.savemat(
        os.path.join('trained_model_weights', 
                     dataset + '_weights_gmm.mat'),
        {'u': u_p.get_value(), 'theta': theta_p.get_value(), 
         'lambda': lambda_p.get_value()})

else:   # pre-train
    assert args.mode == 'pre-train'
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    h = Dense(intermediate_dim[3], activation='relu')(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(Sampling(batch_size, latent_dim), 
               output_shape=(latent_dim,))([z_mean, z_log_var])
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z) 
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-4], activation='relu')(h_decoded)
    x_decoded_mean = Dense(original_dim, activation=datatype)(h_decoded)
    p_c_z = Lambda(get_gamma, output_shape=(n_centroid,))(z_mean)

    ae = Model(x, x_decoded_mean)
    encoder = Model(x, z)    # Only for PreTrainCallback (optional)
    p_c_z_output = Model(x, p_c_z)  
    adam_nn = Adam(lr=lr_nn, epsilon=1e-4)

    ae.compile(optimizer=adam_nn, 
               loss=objectives.binary_crossentropy \
                   if datatype == 'sigmoid' \
                   else objectives.mean_squared_error) 
    
    ae.fit(X, X,
            shuffle=True,
            nb_epoch=epoch,
            batch_size=batch_size,
            callbacks=[PreTrainCallback()])  

    json_str = ae.to_json()
    with open(os.path.join('pretrain_weights', 
                           'ae_' + dataset + '.json'), 'w') as file:
        file.write(json_str)
    ae.save_weights(os.path.join('pretrain_weights', 
                                 'ae_' + dataset + '_weights.h5'))


''' Inference and latent dimension visualization '''
if(inference):
    cluster_labels = p_c_z_output.predict(X,batch_size=batch_size)
    n_samples_plot = 1000
    X = X[0:n_samples_plot]
    sample_z = encoder.predict(X,batch_size=batch_size)
    g = mixture.GaussianMixture(n_components=n_centroid,covariance_type='diag')
    g.fit(sample_z)
    p = g.predict(sample_z)

    z_reduced = TSNE(random_state=RS).fit_transform(sample_z)
    scatter(z_reduced, p, dataset)
    


    # digits generation
    decoder,d_theta,d_u,d_lambda = decoder_init(dataset)
    if (dataset == 'mnist'):
        digit_image = mnist_decoder()
        plt.imshow(digit_image,cmap=cm.gray)
        plt.show()