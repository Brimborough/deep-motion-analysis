import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import AdamTrainer
from nn.Conv2DLayer import Conv2DLayer
from nn.HiddenLayer import HiddenLayer
from nn.Network import Network
from nn.NoiseLayer import NoiseLayer
from nn.Pool2DLayer import Pool2DLayer
from nn.ReshapeLayer import ReshapeLayer

from tools.utils import load_mnist

rng = np.random.RandomState(23455)

datasets = load_mnist(rng)

shared = lambda d: T.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

batchsize = 1

train_set_x = train_set_x.reshape((50000, 1, 28, 28))
valid_set_x = valid_set_x.reshape((10000, 1, 28, 28))
test_set_x  = test_set_x.reshape((10000, 1, 28, 28))

network = Network(
	NoiseLayer(rng, 0.3),

	Conv2DLayer(rng, (4, 1, 5, 5), (batchsize, 1, 28, 28)),
	Pool2DLayer(rng, (batchsize, 4, 28, 28)),
	ActivationLayer(rng, f='ReLU'),
	ReshapeLayer(rng, (4*14*14, )),

	HiddenLayer(rng, (4*14*14, 10)),
	ActivationLayer(rng, f='softmax')
)

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=5, alpha=0.00001, cost='cross_entropy')
trainer.train(network=network, train_input=train_set_x, train_output=train_set_y,
                               valid_input=valid_set_x, valid_output=valid_set_y,
                               test_input=test_set_x, test_output=test_set_y, filename=None)
