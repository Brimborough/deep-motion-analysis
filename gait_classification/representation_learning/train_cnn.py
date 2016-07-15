import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import AdamTrainer, PreTrainer
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
#train_set_x, train_set_y = map(shared, [train_set_x[:5000], train_set_y[:5000]])

train_set_x, train_set_y = map(shared, [train_set_x, train_set_y])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

batchsize = 100

#train_set_x = train_set_x.reshape((5000, 1, 28, 28))
train_set_x = train_set_x.reshape((50000, 1, 28, 28))
valid_set_x = valid_set_x.reshape((10000, 1, 28, 28))
test_set_x  = test_set_x.reshape((10000, 1, 28, 28))

network = Network(
        Conv2DLayer(rng, (4, 1, 5, 5), (batchsize, 1, 28, 28)),
        BatchNormLayer(rng, (batchsize, 4, 28, 28), axes=(0,2,3)),
        ActivationLayer(rng, f='ReLU'),
        Pool2DLayer(rng, (batchsize, 4, 28, 28)),

        ReshapeLayer(rng, (batchsize, 4*14*14)),
        HiddenLayer(rng, (4*14*14, 10)),
        ActivationLayer(rng, f='softmax')
)

#pre_trainer = PreTrainer(rng=rng, batchsize=batchsize, epochs=1, alpha=0.01, cost='mse')
#pre_trainer.pretrain(network=network, pretrain_input=train_set_x, filename=None, logging=False)

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=5, alpha=0.01, cost='cross_entropy')
trainer.train(network=network, train_input=train_set_x, train_output=train_set_y,
                               valid_input=valid_set_x, valid_output=valid_set_y,
                               filename=None, logging=True)

#hyp_optimiser = HyperParamOptimiser(rng=rng, iterations=3)
# Keeping the default values
#hyp_optimiser.set_range()
#hyp_optimiser.optimise(network=network, trainer=trainer,
#                       train_input=train_set_x, train_output=train_set_y,
#                       valid_input=valid_set_x, valid_output=valid_set_y,
#                       filename=None, logging=False)

# Test set performance
trainer.eval(network=network, eval_input=test_set_x, eval_output=test_set_y, filename=None)
