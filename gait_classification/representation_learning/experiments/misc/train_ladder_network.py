import numpy as np
import sys
import theano
import theano.tensor as T

sys.path.append('../..')

from nn.ActivationLayer import ActivationLayer
from nn.AdamTrainer import AdamTrainer
from nn.HiddenLayer import HiddenLayer
from nn.LadderAdamTrainer import *
from nn.LadderNetwork import LadderNetwork

from tools.utils import load_mnist, remove_labels

rng = np.random.RandomState(23455)

datasets = load_mnist(rng)

shared = lambda d: theano.shared(d, borrow=True)

train_set_x, train_set_y = datasets[0]

#train_set_x = shared(train_set_x[:1000])
#train_set_y = train_set_y[:1000]
train_set_x = shared(train_set_x)

# Create a semi-supervised task by removing all but a hundred labels
train_set_y = remove_labels(rng, train_set_y, train_set_y.shape[0] - 100)
train_set_y = shared(train_set_y)


valid_set_x, valid_set_y = map(shared, datasets[1])
#test_set_x, test_set_y = map(shared, datasets[2])

l_train_set_x, u_train_set_x = split_data(train_set_x, train_set_y)
l_train_set_y, u_train_set_y = split_data(train_set_y, train_set_y)

batchsize = 100
sigma = 0.3

network = LadderNetwork(
        encoding_layers =
            [HiddenLayer(rng, (784, 500)),
            ActivationLayer(rng, f='ReLU'), 
            HiddenLayer(rng, (500, 10)),
            ActivationLayer(rng, f='softmax')],

        decoding_layers =
            [HiddenLayer(rng, (500, 10)),
             HiddenLayer(rng, (784, 500))],

        rng=rng,
        # Noise std dev
        sigma=sigma
)

# Weights for the layer-wise unsupervised cost, as found in [1] given bottom-to-top,
# i.e. lambdas[0] is the weight of the reconstruction error for the first
# encoder and last decoder layer
lambdas = np.array([1000., 10., 0.1])

trainer = LadderAdamTrainer(rng=rng, batchsize=batchsize, epochs=10, alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-08,
                            l1_weight=0.0, l2_weight=0.0, supervised_cost='cross_entropy')

trainer.train(network=network, lambdas=lambdas, labeled_train_input = l_train_set_x, labeled_train_output = l_train_set_y,
                                                unlabeled_train_input = u_train_set_x, unlabeled_train_output = u_train_set_y,
                                                valid_input = valid_set_x, valid_output = valid_set_y, filename=None)
