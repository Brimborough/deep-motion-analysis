import numpy as np
import theano
import theano.tensor as T

from nn.AdamTrainer import AdamTrainer
from nn.ActivationLayer import ActivationLayer
from nn.AnimationPlotLines import animation_plot
from nn.DropoutLayer import DropoutLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.VariationalLayer import VariationalLayer
from nn.NoiseLayer import NoiseLayer
from nn.Network import Network, AutoEncodingNetwork, InverseNetwork

from tools.utils import load_cmu, load_cmu_small

rng = np.random.RandomState(23455)

BATCH_SIZE = 1

batchsize = 1
"""
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
"""

network = AutoEncodingNetwork(Network(
    NoiseLayer(rng, 0.3),
    
    Conv1DLayer(rng, (64, 66, 25), (batchsize, 66, 240)),
#    BatchNormLayer(rng, (batchsize, 64, 240), axes=(0,2)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 64, 240)),

    Conv1DLayer(rng, (128, 64, 25), (batchsize, 64, 120)),
#    BatchNormLayer(rng, (batchsize, 128, 120), axes=(0,2)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 128, 120)),
    
    Conv1DLayer(rng, (256, 128, 25), (batchsize, 128, 60)),
#    BatchNormLayer(rng, (batchsize, 256, 60), axes=(0,2)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 256, 60)),

#    ReshapeLayer(rng, shape=(batchsize, 256*30), shape_inv=(batchsize, 256, 30)),
#    HiddenLayer(rng, (256*30, 100)),
#    ActivationLayer(rng, f='ReLU'),
))

shared = lambda d: theano.shared(d, borrow=True)
dataset, std, mean = load_cmu_small(rng)
E = shared(dataset[0][0])

"""
network.load([['../models/cmu/conv_varae/v_0/layer_0.npz', None, None, 
                                        None, '../models/cmu/conv_varae/v_0/layer_1.npz', None, None,],
                                        [None,],
                                        [None, None, '../models/cmu/conv_varae/v_0/layer_3.npz', None,
                                        None, None, '../models/cmu/conv_varae/v_0/layer_4.npz',],])
"""

network.load([None, '../models/cmu/conv_ae/v_0/layer_0.npz', None, None,   # Noise, 1. Conv, Activation, Pooling
                              '../models/cmu/conv_ae/v_0/layer_1.npz', None, None,   # 2. Conv, Activation, Pooling
                              '../models/cmu/conv_ae/v_0/layer_2.npz', None, None,])


def cost(networks, X, Y):
    network_u, network_v, network_d = networks.layers
    
    vari_amount = 0.0
    repr_amount = 1.0
    
    H = network_u(X)
    mu, sg = H[:,0::2], H[:,1::2]
    
    vari_cost = 0.5 * T.mean(mu**2) + 0.5 * T.mean((T.sqrt(T.exp(sg))-1)**2)
    repr_cost = T.mean((network_d(network_v(H)) - Y)**2)
    
    return repr_amount * repr_cost + vari_amount * vari_cost

#trainer = AdamTrainer(rng, batchsize=BATCH_SIZE, epochs=50, alpha=0.00001, cost=cost)

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=50, alpha=0.00001, l1_weight=0.1, l2_weight=0.0, cost='mse')
result = trainer.get_representation(network, E, 2)  * (std + 1e-10) + mean

print result.shape

dataset_ = dataset[0][0] * (std + 1e-10) + mean

new1 = dataset_[460:461]
new2 = result[460:461]
new3 = result[580:581]

animation_plot([new1, new2, new3], interval=15.15)

