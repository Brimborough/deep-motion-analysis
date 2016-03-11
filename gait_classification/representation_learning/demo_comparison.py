import numpy as np
import theano
import theano.tensor as T

from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer

from nn.AnimationPlot import animation_plot

rng = np.random.RandomState(123123)

X = np.load('./data_cmu.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
X = X[:,:-4]

preprocess = np.load('preprocess.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

from network import network
network.load([
    None,
    'layer_0.npz', None, None,
    'layer_1.npz', None, None,
    'layer_2.npz', None, None,
])

for layer in network.layers:
    if isinstance(layer, NoiseLayer): layer.amount = 0.0
    if isinstance(layer, Pool1DLayer):  layer.depooler = lambda x, **kw: x/2

E = theano.shared(X)

while True:

    index = rng.randint(len(X)-1)
    distances = np.inf * np.ones(len(X))
    
    def distance(A, B):
        return (T.mean((A.max(axis=2) - B.max(axis=2))**2) +
                T.mean((A.min(axis=2) - B.min(axis=2))**2))

    tindex = T.lscalar()
    distance_func = theano.function([tindex],
        distance(network(X[index:index+1]), network(E[tindex:tindex+1])))
    
    # Because clips are overlapped, only check every few clips
    for i in range(0, len(X), 4):
        print(i, len(X))
        if i == index: continue
        distances[i] = distance_func(i)
        
    best = np.argsort(distances)
    
    print(index, best[:10])
    
    animation_plot([
        (X[index  :index+1  ] * preprocess['Xstd']) + preprocess['Xmean'],
        (X[best[0]:best[0]+1] * preprocess['Xstd']) + preprocess['Xmean'],
        (X[best[1]:best[1]+1] * preprocess['Xstd']) + preprocess['Xmean']], interval=15.15)
