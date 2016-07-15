import numpy as np
import sys
import theano
import theano.tensor as T

from nn.Ensemble import Ensemble
from nn.Network import Network
from nn.HiddenLayer import HiddenLayer
from nn.ActivationLayer import ActivationLayer
from tools.utils import load_mnist

from copy import deepcopy

rng = np.random.RandomState(23455)

datasets = load_mnist(rng)

shared = lambda d: theano.shared(d, borrow=True)

test_set_x, test_set_y   = map(shared, datasets[2])

batchsize = 100

network1 = Network(
    HiddenLayer(rng, (784, 250)),
    ActivationLayer(rng, f='elu'),

    HiddenLayer(rng, (250, 10)),
    ActivationLayer(rng, f='softmax')
)

network2 = deepcopy(network1)
network3 = deepcopy(network1)
network4 = deepcopy(network1)

networks = [network1, network2, network3, network4]

file_str = 'n%i_layer_%i.npz'

for id, n in enumerate(networks):
    n.load([(file_str % (id+1, 1)), None,
           (file_str % (id+1, 2)), None])

ens = Ensemble(batchsize=batchsize, networks=networks)

ens.eval(test_set_x, test_set_y)
      
