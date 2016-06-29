import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import AdamTrainer
from nn.BatchNormLayer import BatchNormLayer
from nn.Conv2DLayer import Conv2DLayer
from nn.DropoutLayer import DropoutLayer
from nn.HiddenLayer import HiddenLayer
from nn.Network import Network, AutoEncodingNetwork
from nn.NoiseLayer import NoiseLayer
from nn.Pool2DLayer import Pool2DLayer
from nn.ReshapeLayer import ReshapeLayer

from tools.utils import load_dsg

rng = np.random.RandomState(23455)

datasets = load_dsg(rng, (0.8, 0.2)) 
shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
#test_set_x               = shared(datasets[2][0])


batchsize = 100

#network = AutoEncodingNetwork(Network(
#        Conv2DLayer(rng, (10, 3, 3, 3), (batchsize, 3, 64, 64)),
#        BatchNormLayer(rng, (batchsize, 10, 64, 64), axes=(0,2,3)),
#        ActivationLayer(rng, f='ReLU'),
#        Pool2DLayer(rng, (batchsize, 10, 64, 64)),
#
#        Conv2DLayer(rng, (20, 10, 5, 5), (batchsize, 10, 32, 32)),
#        BatchNormLayer(rng, (batchsize, 20, 32, 32), axes=(0,2,3)),
#        ActivationLayer(rng, f='ReLU'),
#        Pool2DLayer(rng, (batchsize, 20, 32, 32)),
#
#        Conv2DLayer(rng, (32, 20, 5, 5), (batchsize, 20, 16, 16)),
#        BatchNormLayer(rng, (batchsize, 32, 16, 16), axes=(0,2,3)),
#        ActivationLayer(rng, f='ReLU'),
#        Pool2DLayer(rng, (batchsize, 32, 16, 16)),
#
#	ReshapeLayer(rng, (batchsize, 32*8*8)),
#	HiddenLayer(rng, (32*8*8, 500)),
#        ActivationLayer(rng, f='ReLU'),
#	HiddenLayer(rng, (500, 10)),
#	ActivationLayer(rng, f='softmax')
#))

network = Network(
        Conv2DLayer(rng, (10, 3, 3, 3), (batchsize, 3, 64, 64)),
        BatchNormLayer(rng, (batchsize, 10, 64, 64), axes=(0,2,3)),
        ActivationLayer(rng, f='ReLU'),
        Pool2DLayer(rng, (batchsize, 10, 64, 64)),

        Conv2DLayer(rng, (20, 10, 5, 5), (batchsize, 10, 32, 32)),
        BatchNormLayer(rng, (batchsize, 20, 32, 32), axes=(0,2,3)),
        ActivationLayer(rng, f='ReLU'),
        Pool2DLayer(rng, (batchsize, 20, 32, 32)),

        Conv2DLayer(rng, (32, 20, 5, 5), (batchsize, 20, 16, 16)),
        BatchNormLayer(rng, (batchsize, 32, 16, 16), axes=(0,2,3)),
        ActivationLayer(rng, f='ReLU'),
        Pool2DLayer(rng, (batchsize, 32, 16, 16)),

	ReshapeLayer(rng, (batchsize, 32*8*8)),
	HiddenLayer(rng, (32*8*8, 500)),
        ActivationLayer(rng, f='ReLU'),
        DropoutLayer(rng, 0.5),
	HiddenLayer(rng, (500, 4)),
	ActivationLayer(rng, f='softmax')
)

network.load([None, '../models/dsg/dae/layer_0.npz', None, None, None, # Noise, 1. Conv, BN, ReLu, Pooling
              '../models/dsg/dae/layer_1.npz', None, None, None, # 2. Conv, BN, ReLu, Pooling 
              '../models/dsg/dae/layer_2.npz', None, None, None, # 3. Conv, BN, ReLu, Pooling
              None, None, None, None, None])                     # Reshape, Hidden, ReLu, Hidden, Softmax

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.001, l1_weight=0.0, l2_weight=1.0, cost='cross_entropy')
trainer.train(network=network, train_input=train_set_x, train_output=train_set_y,
                               valid_input=valid_set_x, valid_output=valid_set_y,
                               test_input=None, test_output=None, 
                               filename=['../models/dsg/class/layer_0.npz', None, None, None, # 1. Conv, BN, ReLu, Pooling
                                         '../models/dsg/class/layer_1.npz', None, None, None, # 2. Conv, BN, ReLu, Pooling 
                                         '../models/dsg/class/layer_2.npz', None, None, None, # 3. Conv, BN, ReLu, Pooling
                                         None, '../models/dsg/class/layer_3.npz', None,       # Reshape, Hidden, ReLu
                                         None, '../models/dsg/class/layer_4.npz', None])      # Dropout, Hidden, Softmax
