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
        extracted='Joe/sequential_final_frame.npz' ,test_start=310, copy_root_velocities=False, control=False, control_type='None', title=None, filename=None):
    
    #Load the preprocessed version, saving on computation
    X = np.load('../../../data/'+orig_file)['clips']
    X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
    X = X[:,:-4]
    preprocess = np.load('../../../data/Joe/preprocess.npz')
    X = (X - preprocess['Xmean']) / preprocess['Xstd']

    # Set if using test set.
    test=True
    data = np.load('../../../data/' + extracted)

    control_sig= np.load('../../../data/Joe/edin_shuffled_control.npz')['control'].swapaxes(1,2)

    if(control_type is 'All'):
        train_control = np.zeros((310,29,24))
        for i in xrange(8,16):
            train_control[:,:,(i-8)*3:((i-8)+1)*3] = np.expand_dims(control_sig[:310,i::8],0)
        test_control = np.zeros((11,29,24))
        for i in xrange(8,16):
            test_control[:,:,(i-8)*3:((i-8)+1)*3] = np.expand_dims(control_sig[310:,i::8],0)
    elif(control_type is 'Alt'):
        train_control = np.zeros((310,29,12))
        for num,i in enumerate([8,12,15]):
            train_control[:,:,num*3:(num+1)*3] = np.expand_dims(control_sig[:310,i::8],0)

        test_control = np.zeros((11,29,12))
        for num,i in enumerate([8,12,15]):
            test_control[:,:,num*3:(num+1)*3]= np.expand_dims(control_sig[310:,i::8],0)
    elif(control_type is 'fl'):
        train_control = np.zeros((310,29,6))
        for num,i in enumerate([8,15]):
            train_control[:,:,num*3:(num+1)*3] = np.expand_dims(control_sig[:310,i::8],0)

        test_control = np.zeros((11,29,6))
        for num,i in enumerate([8,15]):
            test_control[:,:,num*3:(num+1)*3]= np.expand_dims(control_sig[310:,i::8],0)
    else:
        train_control = np.zeros((310,29,9))
        for num,i in enumerate([8,12,15]):
            train_control[:,:,num*3:(num+1)*3] = np.expand_dims(control_sig[:310,i::8],0)

        test_control = np.zeros((11,29,9))
        for num,i in enumerate([8,12,15]):
            test_control[:,:,num*3:(num+1)*3]= np.expand_dims(control_sig[310:,i::8],0)

    if(test):
        data_x = data['test_x']
        data_y = data['test_y']
        data_control = test_control
        # If test data add 310 to the frame
        frame_orig = frame+test_start
    else:
        data_x = data['train_x']
        data_y = data['train_y']
        data_control = train_control
        frame_orig = frame
    
    data_x2 = data['test_x'].copy()

    frames = frame+1
    #Load model
    model.load_weights('../../weights/'+ weight_file)
    pre_lat = np.load('../../../data/' + pre_lstm)

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
    while (30-num_frame_pred) < 30:
        inner_loop_num = 30 - num_frame_pred
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
        data_x = data_loop[:,:,:256].copy() # Copy everything but the final 3 control signals
    else:
        data_x = data_x[:,:,:-3].copy()

    assert not (np.array_equal(data_x, data_x2))
    if(control):
        #orig = orig[:,:,:-3]
        print(data_x.shape)
        print(data_y[:,-1:].shape)
        data_x = np.concatenate((data_x, data_y[:,-1:]), axis=1)

    #check = data_x[:,4:5]
    data_x = (data_x*pre_lat['std']) + pre_lat['mean'] # Sort out the data again, uses final 30
    dat = data_x.swapaxes(2, 1) # Swap back axes
    orig = (orig*pre_lat['std']) + pre_lat['mean']
    orig = orig.swapaxes(2,1)

    X = X[310:]
    print('RMSE between Reconstruction: '+str(rmse(dat,orig)))


    from network import network
    network.load([
        None,
        '../../../models/conv_ae/layer_0.npz', None, None,
        '../../../models/conv_ae/layer_1.npz', None, None,
        '../../../models/conv_ae/layer_2.npz', None, None,
    ])


    def auto(X,orig,dat):
        # Run find_frame.py to find which original motion frame is being used.
        Xorig = X

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
        if(copy_root_velocities):
            Xrecn[:, -3:] = Xorig[:, -3:]
            Xpred[:, -3:] = Xorig[:, -3:]


        #Back to original data space
        Xorig = ((Xorig * preprocess['Xstd']) + preprocess['Xmean'])[:,:,anim_frame_start:anim_frame_end]
        Xrecn = ((Xrecn * preprocess['Xstd']) + preprocess['Xmean'])[:,:,anim_frame_start:anim_frame_end]
        Xpred = ((Xpred * preprocess['Xstd']) + preprocess['Xmean'])[:,:,anim_frame_start:anim_frame_end]

        return Xorig, Xrecn, Xpred

    for frame in [1,2,5,8,10]:
        Xorig,Xrecn,Xpred = auto(X[frame:frame+1], orig[frame:frame+1],dat[frame:frame+1])
        titl = title+" "+str(frame)
        filname = filename+ str(frame) + ".mp4"
        animation_plot([Xorig, Xrecn, Xpred],interval=15.15, labels=['Root','Reconstruction', 'Predicted'], title=titl, filename=filname)

