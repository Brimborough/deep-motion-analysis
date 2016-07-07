import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import AdamTrainer
from nn.Conv1DLayer import Conv1DLayer
from nn.HiddenLayer import HiddenLayer
from nn.Network import Network, AutoEncodingNetwork
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.ReshapeLayer import ReshapeLayer

from tools.utils import load_cmu, load_cmu_small

rng = np.random.RandomState(23455)

shared = lambda d: theano.shared(d, borrow=True)

dataset = load_cmu(rng)

#(17000, 66, 240)
train_set_x_first = shared(dataset[0][0][:, :, :120])
train_set_x_second = shared(dataset[0][0][:, :, 120:])

batchsize = 1
network = AutoEncodingNetwork(Network(
    NoiseLayer(rng, 0.3),
    
    Conv1DLayer(rng, (64, 66, 25), (batchsize, 66, 240)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 64, 240)),

    Conv1DLayer(rng, (128, 64, 25), (batchsize, 64, 120)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 128, 120)),
    
    Conv1DLayer(rng, (256, 128, 25), (batchsize, 128, 60)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 256, 60)),
))

#network.load([None, '../models/cmu/dAe_v_0/layer_0.npz', None, None,   # Noise, 1. Conv, Activation, Pooling
#                    '../models/cmu/dAe_v_0/layer_1.npz', None, None,   # 2. Conv, Activation, Pooling
#                    '../models/cmu/dAe_v_0/layer_2.npz', None, None,]) # 3. Conv, Activation, Pooling
#                    None, '../models/cmu/30062016/layer_3.npz', None])  # Reshape, Hidden, Activation

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=50, alpha=0.00001, l1_weight=0.1, l2_weight=0.0, cost='mse')
trainer.train(network=network, train_input=train_set_x_first, train_output=train_set_x_second, filename=None)
#              filename=[None, '../models/cmu/dAe_v_0/layer_0.npz', None, None,   # Noise, 1. Conv, Activation, Pooling
#                              '../models/cmu/dAe_v_0/layer_1.npz', None, None,   # 2. Conv, Activation, Pooling
#                              '../models/cmu/dAe_v_0/layer_2.npz', None, None,]) # 3. Conv, Activation, Pooling
#                              None, '../models/cmu/dAe_v_0/layer_3.npz', None]) # Reshape, Hidden, Activation
