import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../representation_learning/')

from nn.Conv1DLayer import Conv1DLayer
from nn.Network import Network
from nn.AdamTrainer import AdamTrainer
from nn.ActivationLayer import ActivationLayer
from nn.AnimationPlotLines import animation_plot
from nn.DropoutLayer import DropoutLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.VariationalLayer import VariationalLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork

rng = np.random.RandomState(23455)

#from network import network

"""
network.load([
    None,
    '../models/conv_ae/layer_0.npz', None, None,
    '../models/conv_ae/layer_1.npz', None, None,
    '../models/conv_ae/layer_2.npz', None, None,
])
"""

BATCH_SIZE = 1

network = Network(
    
    Network(
        Conv1DLayer(rng, (64, 66, 25), (BATCH_SIZE, 66, 240)),
        Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240)),
        ActivationLayer(rng),

        DropoutLayer(rng, 0.25),    
        Conv1DLayer(rng, (128, 64, 25), (BATCH_SIZE, 64, 120)),
        Pool1DLayer(rng, (2,), (BATCH_SIZE, 128, 120)),
        ActivationLayer(rng),
    ),
    
    Network(
        VariationalLayer(rng),
    ),
    
    Network(
        InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 120))),
        DropoutLayer(rng, 0.25),    
        Conv1DLayer(rng, (64, 64, 25), (BATCH_SIZE, 64, 120)),
        ActivationLayer(rng),

        InverseNetwork(Pool1DLayer(rng, (2,), (BATCH_SIZE, 64, 240))),
        DropoutLayer(rng, 0.25),    
        Conv1DLayer(rng, (66, 64, 25), (BATCH_SIZE, 64, 240)),
    )
)

network.load([['../models/cmu/conv_varae/v_0/layer_0.npz', None, None, 
                                        None, '../models/cmu/conv_varae/v_0/layer_1.npz', None, None,],
                                        [None,],
                                        [None, None, '../models/cmu/conv_varae/v_0/layer_3.npz', None,
                                        None, None, '../models/cmu/conv_varae/v_0/layer_4.npz',],])

for li, layer in enumerate(network.layers[2].layers):

    if not isinstance(layer, Conv1DLayer): continue

    print(li, layer.W.shape.eval())
    shape = layer.W.shape.eval()
    num = min(shape[0], 64)
    dims = 4, num // 4
    
    if shape[1] < shape[2]:
        dims = dims[1], dims[0]
    
    fig, axarr = plt.subplots(dims[0], dims[1], sharex=False, sharey=False)
    
    W = np.array(layer.W.eval())
    
    for i in range(dims[0]): 
        for j in range(dims[1]):
            axarr[i][j].imshow(
                W[i*dims[1]+j], 
                interpolation='nearest', cmap='rainbow',
                vmin=W.mean() - 5*W.std(), vmax=W.mean() + 5*W.std())
            axarr[i][j].autoscale(False)
            axarr[i][j].grid(False)
            plt.setp(axarr[i][j].get_xticklabels(), visible=False)
            plt.setp(axarr[i][j].get_yticklabels(), visible=False)
    
    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    plt.suptitle('Layer %i Filters' % li, size=16)
    plt.show()
    
