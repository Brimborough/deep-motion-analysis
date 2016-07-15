import numpy as np
import sys
import theano
import theano.tensor as T

sys.path.append('../..')

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import AdamTrainer, PreTrainer
from nn.BatchNormLayer import BatchNormLayer
from nn.Conv1DLayer import Conv1DLayer
from nn.HiddenLayer import HiddenLayer
from nn.Network import Network
from nn.NoiseLayer import NoiseLayer
from nn.Pool1DLayer import Pool1DLayer
from nn.ReshapeLayer import ReshapeLayer

from tools.utils import *

rng = np.random.RandomState(23455)

datasets = load_hdm05_easy_small(rng=rng, split=(0.6, 0.2, 0.2))

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

batchsize = 10

classification_network = Network(
    # Final pooling gives 1, 256, 30 

    # 256*60 = 7680
    ReshapeLayer(rng, (batchsize, 7680)),
    HiddenLayer(rng, (7680, 62)),
    ActivationLayer(rng, f='softmax'),
)

reconstruction_network = InverseNetwork(Network(
    Conv1DLayer(rng, (64, 66, 25), (batchsize, 66, 240)),
    ActivationLayer(rng, f='elu'),
    Pool1DLayer(rng, (2,), (batchsize, 64, 240)),

    Conv1DLayer(rng, (128, 64, 25), (batchsize, 64, 120)),
    ActivationLayer(rng, f='elu'),
    Pool1DLayer(rng, (2,), (batchsize, 128, 120)),
    
    Conv1DLayer(rng, (256, 128, 25), (batchsize, 128, 60)),
    ActivationLayer(rng, f='elu'),
    Pool1DLayer(rng, (2,), (batchsize, 256, 60)),
))

network = Network(
    Conv1DLayer(rng, (64, 66, 25), (batchsize, 66, 240)),
    ActivationLayer(rng, f='elu'),
    Pool1DLayer(rng, (2,), (batchsize, 64, 240)),

    Conv1DLayer(rng, (128, 64, 25), (batchsize, 64, 120)),
    ActivationLayer(rng, f='elu'),
    Pool1DLayer(rng, (2,), (batchsize, 128, 120)),
    
    Conv1DLayer(rng, (256, 128, 25), (batchsize, 128, 60)),
    ActivationLayer(rng, f='elu'),
    Pool1DLayer(rng, (2,), (batchsize, 256, 60)),

    MultiTaskLayer(classification_network, reconstruction_network)
)

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=20, alpha=0.0001, cost='cross_entropy')
trainer.train(network=network, train_input=train_set_x, train_output=train_set_y,
                               valid_input=valid_set_x, valid_output=valid_set_y, filename=None)

trainer.eval(network=network, eval_input=test_set_x, eval_output=test_set_y, filename=None)
