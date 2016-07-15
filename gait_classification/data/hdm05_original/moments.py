import numpy as np
import theano

X = np.load('data_hdm05_original.npz')['clips']

X = X.swapaxes(1, 2)[:,:-4].astype(theano.config.floatX)

Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
Xmean[:,-3:] = 0.0

Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
Xstd[:,-3:-1] = X[:,-3:-1].std()
Xstd[:,-1:  ] = X[:,-1:  ].std()

np.savez_compressed('moments.npz', Xstd=Xstd, Xmean=Xmean)
