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

dataset, std, mean = load_cmu(rng)
train_set_x = shared(dataset[0][0])

batchsize = 1
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

#network.load([None, '../models/cmu/dAe_v_0/layer_0.npz', None, None,   # Noise, 1. Conv, Activation, Pooling
#                    '../models/cmu/dAe_v_0/layer_1.npz', None, None,   # 2. Conv, Activation, Pooling
#                    '../models/cmu/dAe_v_0/layer_2.npz', None, None,]) # 3. Conv, Activation, Pooling
#                    None, '../models/cmu/30062016/layer_3.npz', None])  # Reshape, Hidden, Activation

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=50, alpha=0.00001, l1_weight=0.1, l2_weight=0.0, cost='mse')
trainer.train(network=network, train_input=train_set_x, train_output=train_set_x,
              filename=[None, '../models/cmu/conv_ae/v_0/layer_0.npz', None, None,   # Noise, 1. Conv, Activation, Pooling
                              '../models/cmu/conv_ae/v_0/layer_1.npz', None, None,   # 2. Conv, Activation, Pooling
                              '../models/cmu/conv_ae/v_0/layer_2.npz', None, None,]) # 3. Conv, Activation, Pooling
#                              None, '../models/cmu/dAe_v_0/layer_3.npz', None]) # Reshape, Hidden, Activation

result = trainer.get_representation(network, E, len(network.layers) - 1)  * (std + 1e-10) + mean

print result.shape

dataset_ = dataset[0][0] * (std + 1e-10) + mean

new1 = result[370:371]
new2 = result[470:471]
new3 = result[570:571]

animation_plot([new1, new2, new3], interval=15.15)
