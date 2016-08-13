from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Nadam
from keras.utils.data_utils import get_file
from sklearn import mixture
import numpy as np
import random
import sys
import theano
sys.path.append('../../representation_learning/')
from nn.Network import InverseNetwork, AutoEncodingNetwork
from nn.AnimationPlot import animation_plot

def data_util(preds,x, num_frame_pred):
    print(preds.shape)
    print(x.shape)
    if num_frame_pred>1:
        d2 = np.concatenate((x[:,:(-num_frame_pred)+1], preds), axis=1)
    else:
        d2 = np.concatenate((x, preds),axis=1)
    return d2

def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

def visualise(frame=0 , num_frame_pred=0, anim_frame_start=0, anim_frame_end=240, pre_lstm='Joe/pre_proc_lstm.npz',test_start=310):
    
    preprocess = np.load('../../data/Joe/preprocess.npz')
    
    orig = np.load("orig.npz")['orig']
    skip = np.load("skip.npz")['skip']
    ns = np.load("base.npz")['base']

    skip = skip.swapaxes(2,1)
    

    from network import network
    network.load([
        None,
        '../../models/conv_ae/layer_0.npz', None, None,
        '../../models/conv_ae/layer_1.npz', None, None,
        '../../models/conv_ae/layer_2.npz', None, None,
    ])

    def auto(X,orig):
        # Run find_frame.py to find which original motion frame is being used.
        # Transform dat back to original latent space
        shared3 = theano.shared(X).astype(theano.config.floatX)

        Xorig = InverseNetwork(network)(shared3).eval()
        Xorig = np.array(Xorig)

        # Transform dat back to original latent space
        shared = theano.shared(orig).astype(theano.config.floatX)

        Xrecn = InverseNetwork(network)(shared).eval()
        Xrecn = np.array(Xrecn)

        
        #Back to original data space
        Xorig = ((Xorig * preprocess['Xstd']) + preprocess['Xmean'])[:,:,anim_frame_start:anim_frame_end]
        Xrecn = ((Xrecn * preprocess['Xstd']) + preprocess['Xmean'])[:,:,anim_frame_start:anim_frame_end]

        return Xorig, Xrecn

    title = "Reconstruction vs. ON2 vs. Baseline Sample:"
    filename = "ron2bs"
    for frame in [1,2,5,8,10]:
        Xorig,Xrecn = auto(orig[frame:frame+1], skip[frame:frame+1])
        titl = title+" "+str(frame)
        filname = filename+ str(frame) + ".mp4"
        animation_plot([Xorig, Xrecn, ns[frame:frame+1]],interval=15.15, labels=['Reconstruction', 'ON2', 'Baseline'], title=titl, filename=filname)
visualise()