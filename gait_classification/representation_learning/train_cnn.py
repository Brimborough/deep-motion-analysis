import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import AdamTrainer
from nn.BatchNormLayer import BatchNormLayer
from nn.Conv2DLayer import Conv2DLayer
from nn.HiddenLayer import HiddenLayer
from nn.Network import Network
from nn.NoiseLayer import NoiseLayer
from nn.Pool2DLayer import Pool2DLayer
from nn.ReshapeLayer import ReshapeLayer

from copy import deepcopy
from tools.utils import load_mnist
from tools.HyperParamOptimiser import HyperParamOptimiser

rng = np.random.RandomState(23455)

datasets = load_mnist(rng)

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = datasets[0]
# Keeping training times short (this code is mostly used for debugging)
train_set_x, train_set_y = map(shared, [train_set_x[:1000], train_set_y[:1000]])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

batchsize = 100

#train_set_x = train_set_x.reshape((1000, 1, 28, 28))
#valid_set_x = valid_set_x.reshape((10000, 1, 28, 28))
#test_set_x  = test_set_x.reshape((10000, 1, 28, 28))

#network = Network(
#	Conv2DLayer(rng, (4, 1, 5, 5), (batchsize, 1, 28, 28)),
#        BatchNormLayer(rng, (batchsize, 4, 28, 28), axes=(0,2,3)),
#	ActivationLayer(rng, f='ReLU'),
#	Pool2DLayer(rng, (batchsize, 4, 28, 28)),
#	ReshapeLayer(rng, (batchsize, 4*14*14)),
#	HiddenLayer(rng, (4*14*14, 10)),
#	ActivationLayer(rng, f='softmax')
#)

network = Network(
    HiddenLayer(rng, (784, 500)),
#    BatchNormLayer(rng, (784, 500)),
    ActivationLayer(rng, f='ReLU'),

    HiddenLayer(rng, (500, 10)),
#    BatchNormLayer(rng, (500, 10)),
    ActivationLayer(rng, f='softmax')
)

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=5, alpha=0.01, cost='cross_entropy')

hyp_optimiser = HyperParamOptimiser(rng=rng, iterations=3)
# Keeping the default values
hyp_optimiser.set_range()
hyp_optimiser.optimise(network=network, trainer=trainer,
                       train_input=train_set_x, train_output=train_set_y,
                       valid_input=valid_set_x, valid_output=valid_set_y,
                       test_input=test_set_x, test_output=test_set_y, filename=None, logging=False)
