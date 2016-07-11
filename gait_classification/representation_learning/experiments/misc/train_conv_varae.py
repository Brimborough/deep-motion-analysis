import numpy as np
import theano
import theano.tensor as T

from nn.AdamTrainer import AdamTrainer
from nn.Network import AutoEncodingNetwork

from network_var import network

rng = np.random.RandomState(23455)

#X = np.load('./data_cmu.npz')['clips']
X = np.load('D:/Projects/motioncnndemo/data_cmu.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
X = X[:,:-4]

Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
Xmean[:,-3:] = 0.0

Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
Xstd[:,-3:-1] = X[:,-3:-1].std()
Xstd[:,-1:  ] = X[:,-1:  ].std()

np.savez_compressed('preprocess.npz', Xmean=Xmean, Xstd=Xstd)

X = (X - Xmean) / Xstd

I = np.arange(len(X))
rng.shuffle(I); X = X[I]

E = theano.shared(X, borrow=True)

def cost(networks, X, Y):
    network_u, network_v, network_d = networks.layers
    
    vari_amount = 1.0
    repr_amount = 1.0
    
    H = network_u(X)
    mu, sg = H[:,0::2], H[:,1::2]
    
    vari_cost = 0.5 * T.mean(mu**2) + 0.5 * T.mean((T.sqrt(T.exp(sg))-1)**2)
    repr_cost = T.mean((network_d(network_v(H)) - Y)**2)
    
    return repr_amount * repr_cost + vari_amount * vari_cost

trainer = AdamTrainer(rng, batchsize=1, epochs=25, alpha=0.00001, cost=cost)
trainer.train(network, E, E)
