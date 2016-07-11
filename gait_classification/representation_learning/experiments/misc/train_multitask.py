import numpy as np
import sys
import theano
import theano.tensor as T

sys.path.append('../..')

from nn.ActivationLayer import ActivationLayer
from nn.BinaryTaskTrainer import BinaryTaskTrainer
from nn.BatchNormLayer import BatchNormLayer
from nn.HiddenLayer import HiddenLayer
from nn.MultiTaskLayer import MultiTaskLayer
from nn.Network import MultiTaskNetwork, Network

from tools.utils import load_mnist

rng = np.random.RandomState(23455)

datasets = load_mnist(rng)

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
test_set_x, test_set_y   = map(shared, datasets[2])

classification_network = Network(
    HiddenLayer(rng, (500, 10)),
    ActivationLayer(rng, f='softmax')
)

reconstruction_network = Network(
    HiddenLayer(rng, (500, 784)),
    ActivationLayer(rng, f='ReLU')
)

network = MultiTaskNetwork(
    HiddenLayer(rng, (784, 500)),
    ActivationLayer(rng, f='ReLU'),

    MultiTaskLayer(classification_network, 
                   reconstruction_network)
)

trainer = BinaryTaskTrainer(rng=rng, batchsize=100, epochs=10, alpha=0.01, costs=['cross_entropy', 'mse'])
trainer.train(network=network, train_input=train_set_x, train_outputs=[train_set_y, train_set_x], filename=None)
