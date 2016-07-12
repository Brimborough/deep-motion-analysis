import numpy as np
import sys
import theano
import theano.tensor as T

sys.path.append('../..')

from nn.ActivationLayer import ActivationLayer
from nn.BatchNormLayer import BatchNormLayer
from nn.HiddenLayer import HiddenLayer
from nn.Network import Network
from nn.AdamTrainer import AdamTrainer

from tools.utils import load_mnist

rng = np.random.RandomState(23455)

datasets = load_mnist(rng)

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = map(shared, datasets[0])
valid_set_x, valid_set_y = map(shared, datasets[1])
#test_set_x, test_set_y   = map(shared, datasets[2])

def run_network(alpha = 0.00001, n_hidden = 10):

    network = Network(
        HiddenLayer(rng, (784, n_hidden)),
        BatchNormLayer(rng, (784, n_hidden)),
        ActivationLayer(rng, f='ReLU'),

        HiddenLayer(rng, (n_hidden, 10)),
        BatchNormLayer(rng, (n_hidden, 10)),
        ActivationLayer(rng, f='softmax')
    )

    trainer = AdamTrainer(rng=rng, batchsize=100, epochs=1, alpha=alpha, cost='cross_entropy')
    best_val = trainer.train(network=network, train_input=train_set_x, train_output=train_set_y,
                                              valid_input=valid_set_x, valid_output=valid_set_y, filename=None)

    return best_val

## Called by Spearmint
def main(job_id, params):
    alpha = params['alpha']
    n_hidden = params['n_hidden']

    return run_network(float(alpha), int(n_hidden))
