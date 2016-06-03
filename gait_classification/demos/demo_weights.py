import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../representation_learning/')

from nn.Conv1DLayer import Conv1DLayer

from network import network
network.load([
    None,
    '../models/conv_ae/layer_0.npz', None, None,
    '../models/conv_ae/layer_1.npz', None, None,
    '../models/conv_ae/layer_2.npz', None, None,
])

for li, layer in enumerate(network.layers):

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
    
