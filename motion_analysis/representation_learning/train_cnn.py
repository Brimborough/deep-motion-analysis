import numpy as np
import theano
import theano.tensor as T

from nn.ActivationLayer import ActivationLayer
from nn.HiddenLayer import HiddenLayer
from nn.Conv2DLayer import Conv2DLayer
from nn.Pool2DLayer import Pool2DLayer
from nn.NoiseLayer import NoiseLayer
from nn.BatchNormLayer import BatchNormLayer
from nn.Network import Network
from nn.AdamTrainer import AdamTrainer
from nn.ReshapeLayer import ReshapeLayer

from utils import load_data

rng = np.random.RandomState(23455)

dataset = '../data/mnist/mnist.pkl.gz'
datasets = load_data(dataset)

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

batchsize = 100

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

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=15, alpha=0.0001, cost='cross_entropy')
trainer.train(network=network, train_input=train_set_x, train_output=train_set_y,
                               valid_input=valid_set_x, valid_output=valid_set_y,
                               test_input=test_set_x, test_output=test_set_y, filename=None)
