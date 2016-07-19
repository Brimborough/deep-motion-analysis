from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam
import keras
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import theano

sys.path.append('../../representation_learning/')
from nn.Network import InverseNetwork, AutoEncodingNetwork
from nn.AnimationPlot import animation_plot
# build the model: 2 stacked LSTM

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(TimeDistributed(Dense(256),input_shape=(29,512)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.add(TimeDistributed(Dense(256)))
model.add(Activation(keras.layers.advanced_activations.ELU(alpha=1.0)))
model.compile(loss='mean_squared_error', optimizer='nadam')

frame = 99
num_frame_pred = 20
num_pred_iter = 1

X = np.load('../../data/Joe/edin_shuffled.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
X = X[:,:-4]
preprocess = np.load('../../data/Joe/preprocess.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

# Set if using test set.
data = np.load('../../data/Joe/sequential_final_frame.npz')
train_x = np.concatenate((data['train_x'],data['test_x']))




def data_util(preds,x, num_frame_pred):
    d2 = np.concatenate((x, preds),axis=1)
    return d2


train_y = np.concatenate((data['train_y'],data['test_y']))

train_z = np.zeros((321,29,512))
for i in range(0,28):
	train_z[:,i] = np.concatenate((train_x[:,i], train_x[:,i+1]), axis=1)

data_x = train_z.copy()
data_y = train_y

frame_orig = frame

#Load model
model.load_weights('../weights/base_line.hd5')
pre_lat = np.load('../../data/Joe/pre_proc_lstm.npz')

# To keep shape [1,29,256] allows for easy looping
data_loop = data_x[frame:frame+1]
data_y = data_y[frame:frame+1]
train_t = np.zeros((1,30,512))
# While loop to replace original
 # Replace with zeros to ensure we aren't copying.
data_y[:,(-num_frame_pred)+1:] = 0
while (30-num_frame_pred) < 30:
    preds = model.predict(data_loop) # Predict 29
    if (num_frame_pred != 1):
        preds = preds[:, -num_frame_pred:(-num_frame_pred)+1].copy()
        # Checks to ensure we aren't just copying data
        assert not (np.array_equal(preds, data_x[frame:frame+1, -num_frame_pred:(-num_frame_pred)+1]))
        assert not (np.array_equal(preds, data_loop[:, -num_frame_pred:(-num_frame_pred)+1]))
        # Place prediction into the next location, as predictions is 29 length also, and its the next loc
        data_y[:, (-num_frame_pred)+1:(-num_frame_pred)+2] = preds.copy() 
        print(data_y.shape)
        #Rebuild test
        for i in range(0,28):
    	    train_t[:,i] = np.concatenate((data_y[:,i], data_y[:,i+1]), axis=1)
    	data_loop = train_t.copy()[:, 1:] # Remove the 1st frame so we can loop again
    else:
        preds = preds[:, -num_frame_pred:].copy()
        # Checks to ensure we aren't just copying data
        assert not (np.array_equal(preds, data_x[frame:frame+1, -num_frame_pred:]))
        assert not (np.array_equal(preds, data_loop[:, -num_frame_pred:]))
        data_y[:, -num_frame_pred:] = preds.copy()
        #Rebuild test
        for i in range(0,28):
            print(i)
    	    train_t[:,i] = np.concatenate((data_y[:,i], data_y[:,i+1]), axis=1)
    	data_loop = train_t.copy()[:, 1:] # Remove the 1st frame so we can loop again
    num_frame_pred = num_frame_pred-1


old_preds = data_y[:,-1:]
for it in range(num_pred_iter):
    preds = model.predict(data_loop)[:,-1:] # SHAPE - [1,29,256].
    preds = preds[:,-num_frame_pred:] # Final frame prediction
    """
        Assert:
            that the final data_loop is not equal to the new prediction
            that the final data_loop is equal to the old prediction
    """
    assert not (np.array_equal(preds, data_loop[:,-1:])), "final frame equal to the prediction :S"
    assert (np.array_equal(old_preds, data_y[:, -1:])), "final frame not equal to the old prediction :S"
    data_y = data_util(preds, data_y, num_frame_pred) # concat final frame prediction and data so far
    data_m = data_y #For output
    #Rebuild test
    for i in range(0,29):
    	train_t[:,i] = np.concatenate((data_y[:,i], data_y[:,i+1]), axis=1)
    data_loop = train_t.copy()[:, 1:] # Remove the 1st frame so we can loop again
    data_y = data_y[:, 1:]
    old_preds = preds

data_x = (data_m*pre_lat['std']) + pre_lat['mean'] # Sort out the data again, uses final 30

dat = data_x.swapaxes(2, 1) # Swap back axes

from network import network
network.load([
    None,
    '../../models/conv_ae/layer_0.npz', None, None,
    '../../models/conv_ae/layer_1.npz', None, None,
    '../../models/conv_ae/layer_2.npz', None, None,
])


# Run find_frame.py to find which original motion frame is being used.
Xorig = X[frame_orig:frame_orig+1]

# Transform dat back to original latent space
shared2 = theano.shared(dat).astype(theano.config.floatX)

Xpred = InverseNetwork(network)(shared2).eval()
Xpred = np.array(Xpred) # Just Decoding
#Xrecn = np.array(AutoEncodingNetwork(network)(Xrecn).eval()) # Will the AE help solve noise.

#Xpred[:, -3:] = Xorig[:, -3:]


#Back to original data space
anim_frame_start = 0
anim_frame_end = 240
Xorig = ((Xorig * preprocess['Xstd']) + preprocess['Xmean'])[:,:,anim_frame_start:anim_frame_end]
Xpred = ((Xpred * preprocess['Xstd']) + preprocess['Xmean'])[:,:,anim_frame_start:anim_frame_end]
print(Xpred.shape)
print(Xorig.shape)


animation_plot([Xorig, Xpred], interval=15.15, labels=['Root','Reconstruction', 'Predicted'])
