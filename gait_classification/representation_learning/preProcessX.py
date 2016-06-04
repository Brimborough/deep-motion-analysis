'''
    Simply saving the preprocessed data since it seems to be called everywhere.
'''
import numpy as np
import theano
import sys
sys.path.append('../representation_learning/')


X = np.load('../data/data_cmu.npz')['clips']
preprocess = np.load('../data/Joe/preprocess.npz')

X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
X = X[:,:-4] # - Remove foot contact
X = (X - preprocess['Xmean']) / preprocess['Xstd']

#Save the preprocessed data
np.savez_compressed('../data/Joe/preProcX', preX=X)