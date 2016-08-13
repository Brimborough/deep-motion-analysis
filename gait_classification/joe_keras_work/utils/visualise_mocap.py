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
sys.path.append('../../../representation_learning/')
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

def visualise(model, weight_file, frame=0 , num_frame_pred=1, anim_frame_start=0, anim_frame_end=240, num_pred_iter=10,\
        orig_file='Joe/edin_shuffled.npz', pre_lstm='Joe/pre_proc_lstm.npz',\
        extracted='Joe/sequential_final_frame.npz' ,test_start=310, copy_root_velocities=False, control=False, mixture=False):
    
    # Set if using test set.
    test=True
    data = np.load('../../../data/' + orig_file)['clips']
    data = data[:,:,:-4]
    data_std = data.std()
    data_mean = data.mean(axis=2).mean(axis=0)[np.newaxis, :, np.newaxis]
    data = (data - data_mean) / data_std
    
    control_sig = np.load('../../../data/Joe/edin_shuffled_control.npz')['control'].swapaxes(1,2)

    if(test):
        data_x = data[310:, :-1]
        data_y = data[310:, 1:]
        data_control = control_sig[310:,8::8]
        # If test data add 310 to the frame
        frame_orig = frame+test_start
    else:
        data_x = data[:310, :-1]
        data_y = data[:310, 1:]
        data_control = control_sig[:310,8::8]
        frame_orig = frame
    
    data_x2 = data_x.copy()

    frames = frame+1
    #Load model
    model.load_weights('../../weights/'+ weight_file)

    # Original data set not used in prediction, a check to see what data should look like.
    if num_frame_pred>1:
        orig = np.concatenate([data_x[:,:(-num_frame_pred)+1],data_y[:][:,-num_frame_pred:]], axis=1)
    else:
       orig = np.concatenate([data_x[:],data_y[:, -1:]], axis=1)

    if(control):
        data_x = np.concatenate((data_x, data_control[:,:]), axis=2)
  
    
    # While loop to replace original
    data_loop = data_x[:].copy()
    data_loop2 = data_loop[:].copy()

    # Replace with zeros to ensure we aren't copying.
    data_loop[:,(-num_frame_pred)+1:] = 0
    while (240-num_frame_pred) < 240:
        inner_loop_num = 240 - num_frame_pred
        preds = model.predict(data_loop) # Predict 29
        if (num_frame_pred != 1):
            preds = preds[:, -num_frame_pred:(-num_frame_pred)+1].copy()
            # Checks to ensure we aren't just copying data
            assert not (np.array_equal(preds, data_x[:, -num_frame_pred:(-num_frame_pred)+1]))
            assert not (np.array_equal(preds, data_loop[:, -num_frame_pred:(-num_frame_pred)+1]))
            # Place prediction into the next location, as predictions is 29 length also, and its the next loc
            if(control):
                # Plus one since 29 is the length, not 30
                data_loop[:, (-num_frame_pred)+1:(-num_frame_pred)+2] = np.concatenate((preds.copy(),data_control[:, (-num_frame_pred):(-num_frame_pred)+1]), axis=2) 
            else:
                data_loop[:, (-num_frame_pred)+1:(-num_frame_pred)+2] = preds.copy()
        else:
            preds = preds[:, -num_frame_pred:].copy()
            # Checks to ensure we aren't just copying data
            assert not (np.array_equal(preds, data_x[:, -num_frame_pred:]))
            assert not (np.array_equal(preds, data_loop[:, -num_frame_pred:]))
            if(control):
                data_loop[:, -num_frame_pred:] = np.concatenate((preds.copy(),data_control[:, -num_frame_pred:]), axis=2)
            else:
                data_loop[:, -num_frame_pred:] = preds.copy()
        num_frame_pred = num_frame_pred-1
        
    # Only predict one from now on.
    num_frame_pred = 1
    old_preds = data_loop[:,-num_frame_pred:].copy()
    
    #print(rmse(data_loop[:,:,:-3],orig[:,:-1]))

    for i in range(num_pred_iter):
        preds = model.predict(data_loop) # SHAPE - [frames,29,256].
        preds = preds[:,-1:] # Num_frame_predict.
        """
            Assert:
                that the final data_loop is not equal to the new prediction
                that the final data_loop is equal to the old prediction
        """
        assert not (np.array_equal(preds, data_loop[:,-1:])), "final frame equal to the prediction :S"
        assert (np.array_equal(old_preds, data_loop[:, -1:])), "final frame not equal to the old prediction :S"
        assert not (np.array_equal(preds, data_x[:,-1:])), "Prediction is equal to final data_x"
        # TODO: Make control a LSTM to predict future 
        if(control):
            # Plus one since 29 is the length, not 30
            data_x[:, (-num_frame_pred)+1:(-num_frame_pred)+2] = np.concatenate((preds.copy(),data_control[frame:frames, (-num_frame_pred):(-num_frame_pred)+1]), axis=2) 
        else:
            data_x = data_util(preds, data_loop, num_frame_pred).copy()
        data_loop = data_x[:, 1:].copy()# Remove the 1st frame so we can loop again
        old_preds = preds.copy()
    
    #Final assertion things aren't the same before copying
    assert not (np.array_equal(data_loop, data_loop2))

    if(num_pred_iter == 0): # 0 for ground truth predictions
        data_x = data_loop[:,:,:66].copy() # Copy everything but the final 3 control signals
    else:
        data_x = data_x[:,:,:66].copy()

    assert not (np.array_equal(data_x, data_x2))
    if(control):
        #orig = orig[:,:,:-3]
        data_x = np.concatenate((data_x, data_y[:,-1:]), axis=1)

    #check = data_x[:,4:5]
    data_x = (data_x*data_std) + data_mean # Sort out the data again, uses final 30
    data_x = data_x.swapaxes(2, 1) # Swap back axes
    orig = (orig*data_std) + data_mean
    orig = orig.swapaxes(2,1)
    print(data_x.shape)
    print(orig.shape)

    np.savez_compressed("base", base=data_x)
    
    for frame in [1,2,5,8,10]:
        pred = data_x[frame:frame+1,:, anim_frame_start:anim_frame_end]
        root = orig[frame:frame+1,:, anim_frame_start:anim_frame_end]
        animation_plot([root, pred],interval=15.15, labels=['Root', 'Predicted'])
