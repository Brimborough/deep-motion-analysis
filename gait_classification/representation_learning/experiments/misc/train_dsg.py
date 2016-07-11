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

datasets = load_dsg(rng)
shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x               = shared(datasets[2][0])

batchsize = 100

classification_network = Network(
    ReshapeLayer(rng, shape=(batchsize, 128*8*8)),
    DropoutLayer(rng, 0.5),
    HiddenLayer(rng, (128*8*8, 500)),
    ActivationLayer(rng, f='elu'),
    HiddenLayer(rng, (500, 200)),
    ActivationLayer(rng, f='elu'),
    HiddenLayer(rng, (200, 4)),
    ActivationLayer(rng, f='softmax'),
)

reconstruction_network = InverseNetwork(Network(
    Conv2DLayer(rng, (16, 3, 7, 7), (batchsize, 3, 64, 64)),
    ActivationLayer(rng, f='elu'),
    Pool2DLayer(rng, (batchsize, 16, 64, 64)),

    Conv2DLayer(rng, (64, 16, 5, 5), (batchsize, 16, 32, 32)),
    ActivationLayer(rng, f='elu'),
    Pool2DLayer(rng, (batchsize, 64, 32, 32)),

    Conv2DLayer(rng, (128, 64, 5, 5), (batchsize, 64, 16, 16)),
    ActivationLayer(rng, f='elu'),
    Pool2DLayer(rng, (batchsize, 128, 16, 16)),
))

network = Network(
    Conv2DLayer(rng, (16, 3, 7, 7), (batchsize, 3, 64, 64)),
    BatchNormLayer(rng, (batchsize, 16, 64, 64), axes=(0,2,3)),
    ActivationLayer(rng, f='elu'),
    Pool2DLayer(rng, (batchsize, 16, 64, 64)),

    Conv2DLayer(rng, (64, 16, 5, 5), (batchsize, 16, 32, 32)),
    BatchNormLayer(rng, (batchsize, 64, 32, 32), axes=(0,2,3)),
    ActivationLayer(rng, f='elu'),
    Pool2DLayer(rng, (batchsize, 64, 32, 32)),

    Conv2DLayer(rng, (128, 64, 5, 5), (batchsize, 64, 16, 16)),
    BatchNormLayer(rng, (batchsize, 128, 16, 16), axes=(0,2,3)),
    ActivationLayer(rng, f='elu'),
    Pool2DLayer(rng, (batchsize, 128, 16, 16)),

    MultiTaskLayer(classification_network, reconstruction_network)
#    ReshapeLayer(rng, shape=(batchsize, 128*8*8)),
#    DropoutLayer(rng, 0.5),
#    HiddenLayer(rng, (128*8*8, 500)),
#    ActivationLayer(rng, f='elu'),
#    HiddenLayer(rng, (500, 200)),
#    ActivationLayer(rng, f='elu'),
#    HiddenLayer(rng, (200, 4)),
#    ActivationLayer(rng, f='softmax'),
)

trainer = BinaryTaskTrainer(rng=rng, batchsize=100, epochs=50, alpha=0.001, costs=['cross_entropy', 'mse'])
#trainer = AdamTrainer(rng=rng, batchsize=100, epochs=30, alpha=0.01, cost='cross_entropy')

trainer.train(network=network, train_input=train_set_x, train_outputs=[train_set_y, train_set_x], 
                               valid_input=valid_set_x, valid_output=valid_set_y, 
                               filename=['../models/dsg/layer_1.npz', '../models/dsg/bn_1.npz', None, None,
                                         '../models/dsg/layer_2.npz', '../models/dsg/bn_2.npz', None, None,
                                         '../models/dsg/layer_3.npz', '../models/dsg/bn_3.npz', None, None])

#trainer.train(network=network, train_input=train_set_x, train_output=train_set_y,
#                               valid_input=valid_set_x, valid_output=valid_set_y, filename=None)

#np.savez('network.npz', network=network)
#activations = trainer.get_representation(network, rep_input=test_set_x, depth=len(network.layers)-1)
#print activations.shape
