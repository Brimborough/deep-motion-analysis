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

from nn.AnimationPlotLines import animation_plot

rng = np.random.RandomState(23455)

shared = lambda d: theano.shared(d, borrow=True)

dataset, std, mean = load_cmu_small(rng)

#(17000, 66, 240)
train_set_x_first = shared(dataset[0][0][:, :, :160])
train_set_x_second = shared(dataset[0][0][:, :, 80:])

batchsize = 1

"""
network = AutoEncodingNetwork(Network(
    NoiseLayer(rng, 0.3),
    
    Conv1DLayer(rng, (64, 66, 25), (batchsize, 66, 120)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 64, 120)),

    Conv1DLayer(rng, (128, 64, 25), (batchsize, 64, 60)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 128, 60)),
    
    Conv1DLayer(rng, (256, 128, 25), (batchsize, 128, 30)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 256, 30)),
))
"""

network = AutoEncodingNetwork(Network(
	NoiseLayer(rng, 0.3),

	Conv1DLayer(rng, (64, 66, 25), (batchsize, 66, 160)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 64, 160)),

    Conv1DLayer(rng, (128, 64, 25), (batchsize, 64, 80)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 128, 80)),
    
    Conv1DLayer(rng, (256, 128, 25), (batchsize, 128, 40)),
    ActivationLayer(rng, f='ReLU'),
    Pool1DLayer(rng, (2,), (batchsize, 256, 40)),
))

"""
network.load(['../models/cmu/predict_v_0/layer_0.npz', None, None,   # 1. Conv, Activation, Pooling
             '../models/cmu/predict_v_0/layer_1.npz', None, None,   # 2. Conv, Activation, Pooling
             '../models/cmu/predict_v_0/layer_2.npz', None, None,]) # 3. Conv, Activation, Pooling
"""

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=1, alpha=0.00001, l1_weight=0.1, l2_weight=0.0, cost='mse')
#result = trainer.get_representation(network, train_set_x_first, len(network.network.layers) - 1)  * (std + 1e-10) + mean
#dataset_ = dataset[0][0] * (std + 1e-10) + mean

#new1 = np.concatenate([dataset_[700:701][:, :, :120], result[700:701]], axis=2)
#new2 = dataset_[700:701]

#new1 = new1 * (std + 1e-10) + mean
#new2 = new2 * (std + 1e-10) + mean

#animation_plot([new1, new2], interval=15.15)

trainer.train(network=network, train_input=train_set_x_first, train_output=train_set_x_second,
              filename=[None, '../models/cmu/predict_v_test/layer_0.npz', None, None,   # Noise, 1. Conv, Activation, Pooling
                              '../models/cmu/predict_v_test/layer_1.npz', None, None,   # 2. Conv, Activation, Pooling
                              '../models/cmu/predict_v_test/layer_2.npz', None, None,]) # 3. Conv, Activation, Pooling