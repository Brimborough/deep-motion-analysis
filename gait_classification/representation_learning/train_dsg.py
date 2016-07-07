import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.BinaryTaskTrainer import AdamTrainer, BinaryTaskTrainer
from nn.BatchNormLayer import BatchNormLayer
from nn.Conv2DLayer import Conv2DLayer
from nn.DropoutLayer import DropoutLayer
from nn.HiddenLayer import HiddenLayer
from nn.MultiTaskLayer import MultiTaskLayer
from nn.Network import Network, MultiTaskNetwork, InverseNetwork
from nn.NoiseLayer import NoiseLayer
from nn.Pool2DLayer import Pool2DLayer
from nn.ReshapeLayer import ReshapeLayer

from tools.utils import load_dsg

rng = np.random.RandomState(23455)

datasets = load_dsg(rng, (1.0)) 
shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0][:1000])
#valid_set_x, valid_set_y = map(shared, datasets[1])
#test_set_x               = shared(datasets[2][0])

batchsize = 100

classification_network = Network(
#    DropoutLayer(rng, 0.5),
    HiddenLayer(rng, (500, 4)),
    ActivationLayer(rng, f='softmax')
)

reconstruction_network = InverseNetwork(Network(
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

    ReshapeLayer(rng, shape=(batchsize, 32*8*8), shape_inv=(batchsize, 32, 8, 8)),
    HiddenLayer(rng, (32*8*8, 500)),
    ActivationLayer(rng, f='ReLU'),
))

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

	ReshapeLayer(rng, shape=(batchsize, 32*8*8)),
	HiddenLayer(rng, (32*8*8, 500)),
        ActivationLayer(rng, f='ReLU'),

        MultiTaskLayer(classification_network,
                       reconstruction_network)
)

#network.load([None, '../models/dsg/dae/layer_0.npz', None, None, None, # Noise, 1. Conv, BN, ReLu, Pooling
#              '../models/dsg/dae/layer_1.npz', None, None, None, # 2. Conv, BN, ReLu, Pooling 
#              '../models/dsg/dae/layer_2.npz', None, None, None, # 3. Conv, BN, ReLu, Pooling
#              None, None, None, None, None])                     # Reshape, Hidden, ReLu, Hidden, Softmax

#pre_trainer = PreTrainer(rng=rng, batchsize=batchsize, epochs=1, alpha=0.00001, cost='mse')
#pre_trainer.pretrain(network=network, pretrain_input=train_set_x, filename=None, logging=True)

#trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=1, alpha=0.001, l1_weight=0.0, l2_weight=1.0, cost='cross_entropy')
#trainer.train(network=network, train_input=train_set_x, train_output=train_set_y, filename=None)
#                               filename=['../models/dsg/class/layer_0.npz', None, None, None, # 1. Conv, BN, ReLu, Pooling
#                                         '../models/dsg/class/layer_1.npz', None, None, None, # 2. Conv, BN, ReLu, Pooling 
#                                         '../models/dsg/class/layer_2.npz', None, None, None, # 3. Conv, BN, ReLu, Pooling
#                                         None, '../models/dsg/class/layer_3.npz', None,       # Reshape, Hidden, ReLu
#                                         None, '../models/dsg/class/layer_4.npz', None])      # Dropout, Hidden, Softmax

#trainer = BinaryTaskTrainer(rng=rng, batchsize=100, epochs=10, alpha=0.001, costs=['cross_entropy', 'mse'])
#
#trainer.train(network=network, train_input=train_set_x, train_outputs=[train_set_y, train_set_x], filename=None)
