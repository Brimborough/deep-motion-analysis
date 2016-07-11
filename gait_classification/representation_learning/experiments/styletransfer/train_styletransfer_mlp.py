import numpy as np
import sys
import theano
import theano.tensor as T

sys.path.append('../..')

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import AdamTrainer
from nn.BatchNormLayer import BatchNormLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.DropoutLayer import DropoutLayer
from nn.HiddenLayer import HiddenLayer
from nn.Network import Network
from nn.Pool1DLayer import Pool1DLayer
from nn.ReshapeLayer import ReshapeLayer

from tools.utils import load_styletransfer

rng = np.random.RandomState(23455)

datasets = load_styletransfer(rng=rng, split=(0.6, 0.2, 0.2))

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

train_set_x = train_set_x.reshape((385, 66*240))
valid_set_x = valid_set_x.reshape((87, 66*240))
test_set_x  = test_set_x.reshape((87, 66*240))

batchsize = 10

network = Network(
    HiddenLayer(rng, (66*240, 1000)),
    ActivationLayer(rng, f='ReLU'),

    HiddenLayer(rng, (1000, 500)),
    ActivationLayer(rng, f='ReLU'),

    HiddenLayer(rng, (500, 250)),
    ActivationLayer(rng, f='ReLU'),

    HiddenLayer(rng, (250, 62)),
    ActivationLayer(rng, f='softmax')
)

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.0001, l1_weight=0.0, l2_weight=0.00001, cost='cross_entropy')

trainer.train(network=network, train_input=train_set_x, train_output=train_set_y,
              valid_input=valid_set_x, valid_output=valid_set_y, filename=None)

trainer.eval(network=network, eval_input=test_set_x, eval_output=test_set_y, filename=None)
