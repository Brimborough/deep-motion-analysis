from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Nadam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import theano
sys.path.append('../../../representation_learning/')
from nn.Network import InverseNetwork, AutoEncodingNetwork
from nn.AnimationPlot import animation_plot

def data_util(preds,x, num_frame_pred):
    if num_frame_pred>1:
        d2 = np.concatenate((x[:,:(-num_frame_pred)+1], preds), axis=1)
    else:
        d2 = np.concatenate((x, preds),axis=1)
    return d2

def visualise(model, weight_file, frame=0 , num_frame_pred=1, anim_frame_start=0, anim_frame_end=240, num_pred_iter=10,\
        orig_file='Joe/edin_shuffled.npz', pre_lstm='Joe/pre_proc_lstm.npz', extracted='Joe/sequential_final_frame.npz' ,test_start=310):

    #Load the preprocessed version, saving on computation
    X = np.load('../../../data/'+orig_file)['clips']
    X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
    X = X[:,:-4]
    preprocess = np.load('../../../data/Joe/preprocess.npz')
    X = (X - preprocess['Xmean']) / preprocess['Xstd']

    # Set if using test set.
    test=True
    data = np.load('../../../data/' + extracted)
    print(data.keys())
    if(test):
        data_x = data['test_x']
        data_y = data['test_y']
        # If test data add 310 to the frame
        frame_orig = frame+test_start
    else:
        data_x = data['train_x']
        data_y = data['train_y']
        frame_orig = frame

    #Load model
    model.load_weights('../../weights/'+ weight_file)
    pre_lat = np.load('../../../data/' + pre_lstm)

    # Original data set not used in prediction, a check to see what data should look like.
    if num_frame_pred>1:
        orig = np.concatenate([data_x[frame:frame+1,:(-num_frame_pred)+1],data_y[frame:frame+1][:,-num_frame_pred:]], axis=1)
    else:
       orig = np.concatenate([data_x[frame:frame+1],data_y[frame:frame+1, -1:]], axis=1)
    orig = (orig*pre_lat['std']) + pre_lat['mean']
    orig = orig.swapaxes(2,1)

    # To keep shape [1,29,256] allows for easy looping
    data_loop = data_x[frame:frame+1]
    old_preds = data_loop[:,-1:]
    for i in range(num_pred_iter):
        preds = model.predict(data_loop) # SHAPE - [1,29,256].
        preds = preds[:,-num_frame_pred:] # Final frame prediction
        """
            Assert:
                that the final data_loop is not equal to the new prediction
                that the final data_loop is equal to the old prediction
        """
        assert not (np.array_equal(preds, data_loop[:,-1:])), "final frame equal to the prediction :S"
        assert (np.array_equal(old_preds, data_loop[:, -1:])), "final frame not equal to the old prediction :S"
        data_x = data_util(preds, data_loop, num_frame_pred) # concat final frame prediction and data so far
        data_loop = data_x[:, 1:] # Remove the 1st frame so we can loop again
        old_preds = preds
    
    data_x = (data_x*pre_lat['std']) + pre_lat['mean'] # Sort out the data again, uses final 30

    dat = data_x.swapaxes(2, 1) # Swap back axes

    from network import network
    network.load([
        None,
        '../../../models/conv_ae/layer_0.npz', None, None,
        '../../../models/conv_ae/layer_1.npz', None, None,
        '../../../models/conv_ae/layer_2.npz', None, None,
    ])


    # Run find_frame.py to find which original motion frame is being used.
    Xorig = X[frame_orig:frame_orig+1]

    # Transform dat back to original latent space
    shared = theano.shared(orig).astype(theano.config.floatX)

    Xrecn = InverseNetwork(network)(shared).eval()
    Xrecn = np.array(Xrecn)

    # Transform dat back to original latent space
    shared2 = theano.shared(dat).astype(theano.config.floatX)

    Xpred = InverseNetwork(network)(shared2).eval()
    Xpred = np.array(Xpred) # Just Decoding
    #Xrecn = np.array(AutoEncodingNetwork(network)(Xrecn).eval()) # Will the AE help solve noise.

    # Last 3 - Velocities so similar root
    Xrecn[:, -3:] = Xorig[:, -3:]
    Xpred[:, -3:] = Xorig[:, -3:]


    #Back to original data space
    Xorig = ((Xorig * preprocess['Xstd']) + preprocess['Xmean'])[:,:,anim_frame_start:anim_frame_end]
    Xrecn = ((Xrecn * preprocess['Xstd']) + preprocess['Xmean'])[:,:,anim_frame_start:anim_frame_end]
    Xpred = ((Xpred * preprocess['Xstd']) + preprocess['Xmean'])[:,:,anim_frame_start:anim_frame_end]



    animation_plot([Xorig, Xrecn, Xpred], interval=15.15, labels=['Root','Reconstruction', 'Predicted'])
